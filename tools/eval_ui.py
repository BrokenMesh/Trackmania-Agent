"""Local browser UI for closed-loop test drives: pick a checkpoint and maps, run, stop, restart.

Takeover mode (checkbox): an arrow key while the agent drives (or the Take over button) hands
the car to the player, H (or Hand back) returns it; a stuck / off-track agent waits for the
player. Player segments are saved as training episodes (source "takeover").

    python tools/eval_ui.py --config configs/tmnf.yaml [--port 8600] [--set k=v ...]

Open http://127.0.0.1:8600. One worker thread owns the game connection and the loaded
policy; the page polls /api/status. Results are appended to data/tmnf/eval_ui_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tmagent.config import Config, load_config
from tmagent.data.split import split_of
from tmagent.eval.harness import _json_default, run_episode_sync
from tmagent.eval.progress import ProgressTracker, load_reference
from tmagent.eval.takeover import PlayerSegment, TakeoverControl, run_takeover_episode, save_segment

REPO = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("eval_ui.html")
TRACK_LISTS = ("tmx/tracks.jsonl", "tmx_community/tracks.jsonl")
RESULTS_NAME = "eval_ui_results.jsonl"
CAMPAIGN_RE = re.compile(r"^([A-E])(\d\d)-")
FORMAT_CODE_RE = re.compile(r"\$([0-9a-fA-F]{3}|[lh]\[[^\]]*\]|.)")


class Abort(Exception):
    """Raised from the frame hook to end the running episode early."""


def plain_name(name: str) -> str:
    """TMNF map name without $-format codes ($s, $fff, $l[...] ...); '$$' stays '$'."""
    return FORMAT_CODE_RE.sub(lambda m: "$" if m.group(1) == "$" else "", name).strip()


def list_checkpoints(base: Path) -> list[dict[str, str]]:
    """Training checkpoints under experiments/*/checkpoints, newest run first."""
    out = []
    for run in sorted(base.glob("*/checkpoints"), reverse=True):
        if "-eval-" in run.parent.name:
            continue
        for ckpt in sorted(run.glob("*.pt")):
            rel = ckpt.relative_to(REPO).as_posix()
            out.append({"path": rel, "run": run.parent.name, "file": ckpt.name})
    return out


def list_maps(cfg: Config) -> list[dict[str, Any]]:
    """Downloaded maps that exist in game.map_dir, campaign first, with their data split."""
    d, map_dir = cfg.data, Path(cfg.game.map_dir)
    seen, out = set(), []
    for name in TRACK_LISTS:
        path = Path(d.root) / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            uid = rec["UId"]
            if uid in seen or not (map_dir / f"{uid}.Challenge.Gbx").exists():
                continue
            seen.add(uid)
            title = plain_name(rec.get("TrackName") or uid)
            m = CAMPAIGN_RE.match(title)
            out.append(
                {
                    "uid": uid,
                    "name": title,
                    "campaign": f"{m.group(1)}{m.group(2)}" if m else None,
                    "split": split_of(uid, d.split_seed, d.val_frac, d.test_frac),
                }
            )
    out.sort(key=lambda m: (m["campaign"] is None, m["campaign"] or "", m["name"].lower()))
    return out


@dataclass
class Job:
    ckpt: str
    uid: str
    name: str
    seed: int
    takeover: bool = False


@dataclass
class Runner:
    """Owns the game and the policy; runs queued jobs one at a time on a worker thread."""

    cfg: Config
    results_path: Path
    queue: deque[Job] = field(default_factory=deque)
    current: dict[str, Any] | None = None
    error: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.control = TakeoverControl()
        self.saved_segments = 0
        self.abort_reason: str | None = None
        self.game = None
        self.policy = None
        self.policy_path: str | None = None
        if self.results_path.exists():
            for line in self.results_path.read_text(encoding="utf-8").splitlines():
                self.results.append(json.loads(line))
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(
        self, ckpt: str, maps: list[dict[str, str]], repeats: int, takeover: bool = False
    ) -> None:
        with self.lock:
            for m in maps:
                for seed in range(repeats):
                    self.queue.append(Job(ckpt, m["uid"], m["name"], seed, takeover))
            self.error = None
        self.wake.set()

    def stop(self) -> None:
        """Clear the queue and abort the running episode."""
        with self.lock:
            self.queue.clear()
            self.abort_reason = "stopped"

    def restart(self) -> None:
        """Abort the running episode and run it again from the start."""
        with self.lock:
            if self.current is not None:
                job = self.current["job"]
                self.queue.appendleft(job)
                self.abort_reason = "restarted"

    def takeover_request(self, what: str) -> None:
        """ "take_over", "hand_back" or "retry" for the running takeover episode."""
        self.control.request(what)

    def skip(self) -> None:
        with self.lock:
            if self.current is not None:
                self.abort_reason = "skipped"

    def clear_results(self) -> None:
        with self.lock:
            self.results.clear()
            self.results_path.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        with self.lock:
            cur = None
            if self.current is not None:
                job = self.current["job"]
                cur = {**{k: v for k, v in self.current.items() if k != "job"}, **job.__dict__}
                cur["phase"] = self.control.phase if job.takeover else "agent"
            return {
                "current": cur,
                "queue": [j.__dict__ for j in self.queue],
                "error": self.error,
                "saved_segments": self.saved_segments,
                "results": self.results[-200:],
            }

    def _loop(self) -> None:
        while True:
            self.wake.wait()
            with self.lock:
                if not self.queue:
                    self.wake.clear()
                    continue
                job = self.queue.popleft()
                self.abort_reason = None
                self.current = {
                    "job": job,
                    "progress": 0.0,
                    "race_time_ms": 0,
                    "started": time.time(),
                }
            try:
                result = self._run(job)
            except Abort as exc:
                result = {"aborted": str(exc)}
                self._settle_game()
            except Exception as exc:
                traceback.print_exc()
                self._drop_game()
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.queue.clear()
                result = {"aborted": "error"}
            with self.lock:
                self.current = None
                if result.get("aborted") != "restarted":
                    self._record(job, result)

    def _run(self, job: Job) -> dict[str, Any]:
        policy = self._policy(job.ckpt)
        game = self._game()
        ref = load_reference(job.uid, self.cfg.data.root)
        if not job.takeover:
            return run_episode_sync(
                game, policy, job.uid, self.cfg, job.seed, ref, on_frame=self._on_frame
            )
        self.control.pop()
        return run_takeover_episode(
            game, policy, job.uid, self.cfg, job.seed, ref, self.control,
            on_frame=self._on_frame, on_segment=lambda seg: self._save_segment(job, seg),
        )  # fmt: skip

    def _save_segment(self, job: Job, seg: PlayerSegment) -> None:
        if save_segment(seg, self.cfg, job.uid, job.name, job.ckpt) is not None:
            with self.lock:
                self.saved_segments += 1

    def _settle_game(self) -> None:
        """After an abort while the player drove: back to sync, so the plugin pauses again."""
        if self.control.phase != "agent" and self.game is not None:
            self.game.take_back()
        self.control.phase = "agent"

    def _on_frame(self, tracker: ProgressTracker) -> None:
        with self.lock:
            if self.current is not None:
                self.current["progress"] = tracker.progress
                self.current["race_time_ms"] = tracker.time_ms
            if self.abort_reason is not None:
                raise Abort(self.abort_reason)

    def _policy(self, ckpt: str):
        if self.policy_path != ckpt:
            from tmagent.model.streaming import load_streaming_policy

            self.policy = None
            self.policy = load_streaming_policy(REPO / ckpt, self.cfg.runtime.device)
            self.policy_path = ckpt
        return self.policy

    def _game(self):
        if self.game is None:
            from tmagent.game import make_game

            self.game = make_game(self.cfg.game, self.cfg.data)
        return self.game

    def _drop_game(self) -> None:
        if self.game is not None:
            try:
                self.game.close()
            except Exception:
                pass
        self.game = None

    def _record(self, job: Job, result: dict[str, Any]) -> None:
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt": job.ckpt,
            "name": job.name,
            **result,
        }
        rec["map"], rec["seed"] = job.uid, job.seed
        self.results.append(rec)
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def make_handler(runner: Runner, options: dict[str, Any]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(json.dumps(obj, default=_json_default).encode(), "application/json", code)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/options":
                self._json(options)
            elif self.path == "/api/status":
                self._json(runner.status())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/run":
                known = {m["uid"]: m for m in options["maps"]}
                maps = [known[u] for u in body.get("maps", []) if u in known]
                ckpts = {c["path"] for c in options["checkpoints"]}
                if body.get("ckpt") not in ckpts or not maps:
                    self._json({"error": "pick a model and at least one map"}, 400)
                    return
                repeats = max(1, min(int(body.get("repeats", 1)), 10))
                runner.submit(body["ckpt"], maps, repeats, bool(body.get("takeover")))
            elif self.path == "/api/stop":
                runner.stop()
            elif self.path == "/api/restart":
                runner.restart()
            elif self.path == "/api/skip":
                runner.skip()
            elif self.path in ("/api/take_over", "/api/hand_back", "/api/retry"):
                runner.takeover_request(self.path.rsplit("/", 1)[1])
            elif self.path == "/api/clear":
                runner.clear_results()
            else:
                self._json({"error": "not found"}, 404)
                return
            self._json({"ok": True})

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Browser UI for closed-loop test drives.")
    ap.add_argument("--config", required=True, help="YAML config (game, data, eval sections)")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--default-ckpt", default=None, help="checkpoint preselected in the page")
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="K=V"
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    options = {
        "checkpoints": list_checkpoints(REPO / "experiments"),
        "maps": list_maps(cfg),
        "default_ckpt": args.default_ckpt,
    }
    runner = Runner(cfg, Path(cfg.data.root) / RESULTS_NAME)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner, options))
    print(
        f"eval UI on http://127.0.0.1:{args.port} ({len(options['maps'])} maps, "
        f"{len(options['checkpoints'])} checkpoints)"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
