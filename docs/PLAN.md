# Projekt: Trackmania-Agent mit Video-Kontext und Aktions-Historie

## Ziel

Ein Modell mit höchstens 3B Parametern, das Trackmania live spielt.

- **Eingabe:** Ringpuffer mit den letzten Frames (Video-Historie) und den eigenen letzten Aktionen.
- **Ausgabe:** Aktions-Chunk (Lenkung, Gas, Bremse) für die nächsten Schritte.
- **Steuerfrequenz:** 60 Hz im Live-Betrieb, auf der lokalen GPU des Nutzers.
- **Training:** Behavior Cloning auf Replays, anschliessend Verbesserung durch eigene Läufe.

Messbare Erfolgskriterien (Zahlen in Phase 0 festlegen und hier eintragen):

- Finish-Rate auf Held-out-Strecken (Split nach Strecke, nicht nach Replay): _TBD_
- Streckenfortschritt in % im Median: _TBD_
- Zeit relativ zur Referenz-Fahrt auf denselben Strecken: _TBD_
- Verpasste 60-Hz-Deadlines im Live-Betrieb: unter 1 %

## Architektur-Entscheidungen

- **"60 fps" heisst: Steuerschleife mit 60 Hz.** Das Modell läuft nicht pro Frame. Es sagt Chunks von z. B. 8 Aktionen voraus (ca. 133 ms) und läuft asynchron zur Steuerschleife. Wahrnehmung mit 20-30 fps reicht.
- **Kontext:** Frames als Tokens (Pooling auf 16-64 Tokens pro Frame), dazu eine Aktions-Token pro Schritt, zeitlich verschachtelt, kausaler Transformer, KV-Cache im Streaming-Betrieb. Start mit 2 s Historie, später 8-30 s.
- **Aktionen als Eingabe:** Gewünscht. Bekanntes Risiko: Das Modell kopiert die letzte Aktion und ignoriert das Bild (Copycat-Problem). Gegenmassnahmen: Action-History-Dropout, Rauschen auf Eingabe-Aktionen, Ablation mit und ohne Aktions-Historie. Entscheidung anhand der Closed-Loop-Metriken, nicht des Trainingsverlusts.
- **Backbone:** Kleiner Vision-Encoder (SigLIP2-Klasse) plus Transformer über die Zeit. Aktions-Kopf zuerst einfach (Regression/Diskretisierung), Flow-Matching nur wenn nötig. Lenkung ist niedrigdimensional.
- **Grösse:** Start bei 0,3-0,5B zur Validierung der Pipeline. Hochskalieren Richtung 3B nur, wenn das Latenzbudget es erlaubt. Ausgangspunkt dürfen offene Gewichte sein (z. B. NitroGen, GR00T N1). Das Code-Repo muss die Lizenzen vorher prüfen.

## Phasen

### Phase 0: Machbarkeit (zuerst, vor jedem Training)

1. GPU, VRAM und CUDA-Version erfassen (`nvidia-smi`). Dokumentieren, welche Modellgrösse beim Training und bei der Inferenz hineinpasst.
2. Trackmania-Version und Schnittstellen klären: Replay-Format, Möglichkeit, Replays im Spiel abzuspielen und zu rendern, Bildschirmerfassung, virtuelles Gamepad (z. B. Openplanet, TMInterface, vgamepad). Alle Annahmen selbst verifizieren, nichts aus Erinnerung übernehmen.
3. Nutzungsbedingungen von Spiel und Replay-Quellen prüfen, bevor Daten in grösserem Umfang heruntergeladen oder das Spiel automatisiert wird. Bei Unklarheit stoppen und den Nutzer fragen.
4. Latenz der Kette messen: Bildschirmerfassung → Vorverarbeitung → Modell → virtuelles Gamepad. Ergebnis in `docs/latency.md`.
5. Latenzbudget aufstellen und daraus Chunk-Länge und maximale Modellgrösse ableiten.

### Phase 1: Datenpipeline

- Replays laden, im Spiel mit fester Kamera, festem HUD und fester Auflösung rendern, Frames und Eingaben exakt synchronisieren und auf 60 Hz resampeln.
- Speicherformat: Shards mit komprimierten Frames (z. B. 224x224 oder kleiner) plus Aktionsarrays, Metadaten (Strecke, Fahrer, Zeit).
- Split nach Strecke: Trainings-, Validierungs- und Test-Strecken.
- Qualitätschecks: Synchronität (Frame t gehört zu Aktion t), Zeitstempel, Duplikate.
- Replays unterschiedlicher Fahrerqualität einbeziehen, nicht nur Bestzeiten.

### Phase 2: Baseline

- Einzelbild-Behavior-Cloning (NitroGen-Muster) als erste lauffähige Version.
- Closed-Loop-Evaluierungs-Harness bauen: Modell fährt live, Metriken werden geloggt (Fortschritt, Finish, Zeit, Abweichungen von der Strecke).
- Ohne diese Harness wird kein weiteres Training begonnen.

### Phase 3: Kontext-Modell

- Frame-Historie und Aktions-Puffer einbauen.
- Ablationen: Länge der Historie (0, 1 s, 2 s, 8 s), mit und ohne Aktions-Historie, mit und ohne Dropout.
- Jede Variante mit derselben Evaluierung und mehreren Seeds vergleichen.

### Phase 4: Echtzeit

- Action-Chunking, asynchrone Inferenz, KV-Cache-Streaming, bf16 und `torch.compile`, bei Bedarf TensorRT.
- Latenzprofil pro Komponente, p50/p99 loggen.
- Die 60-Hz-Steuerschleife muss die Aktionen aus dem letzten Chunk interpolieren und darf nie auf das Modell warten.

### Phase 5: Closed-Loop-Verbesserung

- Eigene Läufe aufzeichnen, Fehlerzustände durch Experten-Aktionen oder Replays nachlabeln (DAgger-artig), neu trainieren.
- Rauschen auf Aktionen während der Datenerzeugung, um Erholung von Abweichungen zu lernen.
- Reinforcement Learning ist ausdrücklich ausserhalb des Umfangs, ausser der Nutzer fordert es.

### Phase 6: Skalierung (optional)

- Erst wenn Phase 4 stabil läuft und das Latenzbudget Reserve zeigt: grösserer Backbone bis maximal 3B.
- Vergleich zum kleinen Modell mit gleichen Metriken. Behalten, was messbar besser ist.

## Arbeitsregeln für Claude Code

- Kleine, überprüfbare Schritte. Nach jedem Schritt Tests oder Messung.
- Konfiguration über Dateien (YAML), keine harten Pfade. Alle Experimente mit Config, Seed, Git-Hash und Metriken unter `experiments/<datum>-<name>/` ablegen.
- Vor jedem GPU-Lauf länger als 30 Minuten: Dauer schätzen und den Nutzer informieren.
- Keine grossen Downloads und keine Spielautomatisierung ohne Prüfung der Nutzungsbedingungen (Phase 0, Punkt 3).
- Fehlschläge und Entscheidungen in `docs/decisions.md` festhalten (was probiert, was gemessen, was verworfen).
- Zahlen nur aus eigenen Messungen übernehmen. Aussagen zu Tools, APIs und Lizenzen vor der Nutzung gegen die Dokumentation prüfen.

## Bekannte Risiken

- Verteilungsverschiebung: Replays zeigen fast nur gute Linien, das Modell sieht eigene Fehler selten (Phase 5 adressiert das).
- Copycat-Verhalten durch die Aktions-Historie (Phase 3 testet das).
- Latenz: Ein 3B-Modell passt womöglich nicht in 60-Hz-Chunks auf der vorhandenen GPU. Dann bleibt die Obergrenze unter 3B.
- Synchronisationsfehler zwischen Frame und Aktion verschlechtern das Training, ohne dass es auffällt. Daher Qualitätschecks in Phase 1.
- Rechtliche Lage von Replay-Nutzung und Automatisierung ist ungeprüft.
