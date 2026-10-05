/*
 * TMAgentLink - TMInterface 2.x plugin: TCP bridge between TrackMania Nations Forever
 * and the tmagent Python client (tmagent/game/tmnf/client.py).
 *
 * Wire protocol: tmagent/game/tmnf/PROTOCOL.md (version 1). Both sides are written
 * by hand from that document; keep PROTOCOL_VERSION in sync with protocol.py.
 * Original code written for tmagent. API names were confirmed against the public
 * sources listed below, no third-party plugin code is copied (docs/decisions.md D-009).
 *
 * ---------------------------------------------------------------------------------
 * INSTALLATION (Windows, TMNF + TrackMania ModLoader with the TMInterface 2.x mod)
 * ---------------------------------------------------------------------------------
 *  1. Copy this file to  Documents\TMInterface\Plugins\TMAgentLink.as
 *  2. Start TMNF through TMLoader (profile with the TMInterface mod). Optionally set
 *     the port at launch:   /configstring="set tmagent_port 8477"
 *     (default 8477, must equal game.tmi_port in the tmagent config). In the TMInterface
 *     console (command toggle_console) you can also type: set tmagent_port 8477
 *     and then: tmagent_listen
 *  3. Enable the plugin in the TMInterface window (Settings -> Plugins tab, tick
 *     "TMAgentLink"; exact menu names UNVERIFIED, see docs/setup_windows.md). The console
 *     log must show "TMAgentLink listening on 127.0.0.1:<port>". The console command
 *     `tmagent_status` prints the current session.
 *  4. Run tools/tmnf_smoke.py on the Python side (docs/setup_windows.md).
 *
 * ---------------------------------------------------------------------------------
 * API NAMES RELIED ON, with their source. Status:
 *   VERIFIED   = seen in working third-party TMI 2.x plugin code AND/OR in the API
 *                declarations (as.predefined, TMNF folder of github.com/sashi0034/angel-lsp)
 *   UNVERIFIED = declared but runtime behaviour not confirmed on TMNF; first-run check
 * Every UNVERIFIED call is isolated in one Api*() helper (section 2): fix it there.
 *
 *  VERIFIED   Net::Socket: Listen(host, port), Accept(0) in Render(), Available,
 *             ReadInt32/ReadUint8/ReadFloat/ReadString(n)/ReadBytes(n), Write(int|uint8|
 *             float|string|array<uint8>) -> bool, NoDelay, RemoteIP
 *             [Linesight Python_Link.as; donadigo gist c010cd68...; Archmetrus/TMNf-RLAgent
 *             RealtimeDataPublisher.as (TMNF, non-blocking Available polling in OnRunStep)]
 *  VERIFIED   Callbacks: Main(), Render(), OnRunStep(SimulationManager@),
 *             OnCheckpointCountChanged(SimulationManager@, int current, int target),
 *             OnGameStateChanged(TM::GameState), OnDisabled(), GetPluginInfo()
 *             [Linesight; Sai-Moen / XD1674 plugins]
 *  VERIFIED   simManager.RaceTime (negative during the countdown), .InRace, .SetSpeed(float)
 *             (0 allowed), .SetInputState(InputType::Left|Right|Up|Down, 0|1), .SaveState(),
 *             .RewindToState(state), .GiveUp(), .PreventSimulationFinish(),
 *             .PlayerInfo.RaceFinished, .Dyna.CurrentState.Location.Position,
 *             .Dyna.CurrentState.LinearSpeed (m/s, km/h = Length()*3.6)
 *             [Linesight; Archmetrus RealtimeDataPublisher.as; XD1674 velocity_bf_v2.as]
 *  VERIFIED   ExecuteCommand("map <path>"), GetCurrentGameState(), RegisterVariable,
 *             GetVariableDouble, RegisterCustomCommand, CommandList.Process
 *             [Linesight game_instance_manager.py / Python_Link.as]
 *  VERIFIED   Graphics::CaptureScreenshot(vec2(w, h)) inside Render(): array<uint8>
 *             of w*h*4 bytes, BGRA [Linesight Python_Link.as + python side (BGRA2GRAY)].
 *             Row order and "does it show the latest simulated tick" are UNVERIFIED
 *             (client.CAPTURE_FLIP_VERTICAL, REQUEST_FRAME settle)
 *  VERIFIED   SetInputState / SetSpeed called from Render()-time message handling
 *             (Linesight does both while answering a frame request)
 *  UNVERIFIED InputType::Steer with SetInputState(Steer, int) on TMNF (enum is declared;
 *             D-010 keeps binary steering as the default)               -> ApiSetSteer
 *  UNVERIFIED SimulationState.Dyna.CurrentState on a SaveState() snapshot -> ApiSavedPose
 *             (Linesight reads the same fields on the Python side of a saved state)
 *  UNVERIFIED GetCurrentChallenge().Uid / .Name (declared in as.predefined)  -> ApiMapInfo
 *  UNVERIFIED that SetSpeed(0) stops ticks immediately. Two guards keep the held state
 *             exact if it does not: OnRunStep rewinds to it while paused, and Render()
 *             (CheckPausedDrift) rewinds if the race time differs from the held one.
 *             RewindToState is VERIFIED from OnRunStep (Linesight), UNVERIFIED from Render()
 *  UNVERIFIED that OnRunStep is called at race time 0 (Linesight saves its start state
 *             there). If the first callback after the countdown has race time > 0, STATE
 *             replies after RESTART report that time and the client raises a clear error.
 *             RaceTime right after RewindToState is not read: the saved state's time is reported
 *  UNVERIFIED whether OnRunStep is invoked once or twice for the tick we paused in;
 *             STEP targets are race-time based (paused time + 10 * n) so both work
 *  UNVERIFIED `map` argument format (absolute vs relative to Tracks\Challenges)
 *             -> the Python side chooses the string, see game.map_command_path
 *
 * ---------------------------------------------------------------------------------
 * TESTING
 * ---------------------------------------------------------------------------------
 *  Compile-checked with AngelScript 2.35.1 against stubs of the declared TMI API and run
 *  against a simulated engine (tick order, late SetSpeed, rewind lag, missing events) with
 *  the real Python client. That proves the protocol and the state machine, not the real
 *  engine: tools/tmnf_smoke.py is the first-run check (step additivity, determinism).
 *  TMI embeds its own AngelScript version; syntax errors would show in the TMI log.
 *
 * ---------------------------------------------------------------------------------
 * WHEN THIS PLUGIN BLOCKS (details and rationale: PROTOCOL.md "Blocking model")
 * ---------------------------------------------------------------------------------
 *  - Commands are read with NON-BLOCKING polls (Available >= 8) in Render() and in
 *    OnRunStep(). The game thread never waits for the client.
 *  - Exception 1: once a message header has arrived, the rest of that message is awaited
 *    for at most PAYLOAD_TIMEOUT_MS (a message is one sendall on the client side).
 *  - Exception 2: Write() may block if the client stops reading (TCP buffer full).
 *  - "Paused" in sync mode = SetSpeed(0), not a blocked thread, so Render() keeps running
 *    and screenshots can be taken. A command sent while paused is seen in the next Render().
 */

// ================================================================== 1. constants

const int PROTOCOL_VERSION = 1;
const string PLUGIN_NAME = "TMAgentLink";
const string PLUGIN_VERSION = "0.1.0";
const string HOST = "127.0.0.1";
const int DEFAULT_PORT = 8477;
const uint HEADER_SIZE = 8;
const uint MAX_COMMAND_PAYLOAD = 1048576;   // 1 MiB: commands are tiny, bigger = corruption
const uint PAYLOAD_TIMEOUT_MS = 2000;
const int MAX_MESSAGES_PER_POLL = 32;
const int TICK_MS = 10;
const int PIXEL_BGRA8 = 0;
const uint LOAD_GAP_MS = 250;               // no OnRunStep for this long = a map is loading
const uint LISTEN_FALLBACK_MS = 5000;       // listen even if the deferred command never ran

// client -> plugin
const int CMD_HELLO = 1;
const int CMD_SET_MODE = 2;
const int CMD_LOAD_MAP = 3;
const int CMD_RESTART = 4;
const int CMD_STEP = 5;
const int CMD_SET_INPUT = 6;
const int CMD_REQUEST_FRAME = 7;
const int CMD_STREAM_FRAMES = 8;
const int CMD_SET_SPEED = 9;
const int CMD_EXECUTE = 10;
const int CMD_GET_STATE = 11;
const int CMD_CLOSE = 12;
const int CMD_STREAM_STATE = 13;
const int CMD_PING = 14;

// plugin -> client
const int REP_ACK = 101;
const int REP_ERROR = 102;
const int REP_STATE = 103;
const int REP_FRAME = 104;
const int REP_HELLO = 105;
const int REP_PUSH_STATE = 106;
const int REP_PUSH_FRAME = 107;

const int RESTART_REWIND = 0;
const int RESTART_GIVE_UP = 1;

// pending (deferred) operation, completed from OnRunStep / OnCheckpointCountChanged
const int OP_NONE = 0;
const int OP_LOAD_WAIT = 1;     // waiting for race time 0 of a new race (LOAD_MAP / give-up RESTART)
const int OP_STEP = 2;          // running until the target race time
const int OP_RESTART = 3;       // rewind to the start state at the next OnRunStep
const int OP_PAUSE = 4;         // SET_MODE sync while a race runs: pause at the next OnRunStep

// ================================================================== 2. API shims
// One helper per game API that is not fully verified. A first-run fix is a one-line edit.
// Null checks: outside a race (menus, loading) the pose/player objects may not exist.

SimulationManager@ g_sim = GetSimulationManager();

// VERIFIED (Linesight, Archmetrus, Sai-Moen): negative during the countdown.
int ApiRaceTime(SimulationManager@ sm) { return sm.RaceTime; }

// VERIFIED (Linesight). Whether it is already true during the countdown is UNVERIFIED.
bool ApiInRace(SimulationManager@ sm) { return sm.InRace; }

// Finished = checkpoint callback saw current == target, or the player info says so.
// VERIFIED property (Linesight CRaceFinished); Linesight also treats TickTime > RaceTime
// as finished, not used here because its semantics are unclear.
bool ApiPlayerFinished(SimulationManager@ sm)
{
    TM::PlayerInfo@ pi = sm.PlayerInfo;
    return pi !is null && pi.RaceFinished;
}

// VERIFIED (Linesight): 0 pauses, called from OnRunStep and from Render()-time handlers.
void ApiSetSpeed(SimulationManager@ sm, float speed) { sm.SetSpeed(speed); }

// VERIFIED (Linesight): SaveState at race time 0, RewindToState(state) later.
SimulationState@ ApiSaveState(SimulationManager@ sm) { return sm.SaveState(); }

void ApiRewind(SimulationManager@ sm, SimulationState@ st) { sm.RewindToState(st); }

// VERIFIED (Linesight): the game's own restart (countdown follows).
void ApiGiveUp(SimulationManager@ sm) { sm.GiveUp(); }

// VERIFIED (Linesight calls it from OnCheckpointCountChanged when current == target).
void ApiPreventFinish(SimulationManager@ sm) { sm.PreventSimulationFinish(); }

// UNVERIFIED: analog steering on TMNF (InputType::Steer is declared; D-010 keeps binary
// steering as the default). Value is the TMI steer int, negative = left.
void ApiSetSteer(SimulationManager@ sm, int steer) { sm.SetInputState(InputType::Steer, steer); }

// UNVERIFIED: Dyna of a saved state (used for STATE replies while paused). Linesight reads
// the same fields from a saved state on the Python side.
void ApiSavedPose(SimulationState@ st, vec3 &out pos, vec3 &out vel)
{
    TM::HmsDyna@ d = st.Dyna;
    if (d is null) return;
    pos = d.CurrentState.Location.Position;
    vel = d.CurrentState.LinearSpeed;
}

// VERIFIED (Archmetrus, XD1674): live position and velocity (m/s).
void ApiLivePose(SimulationManager@ sm, vec3 &out pos, vec3 &out vel)
{
    TM::HmsDyna@ d = sm.Dyna;
    if (d is null) return;
    pos = d.CurrentState.Location.Position;
    vel = d.CurrentState.LinearSpeed;
}

// UNVERIFIED: "uid<TAB>name" of the loaded map (LOAD_MAP reply text); declared in as.predefined.
string ApiMapInfo()
{
    TM::GameCtnChallenge@ ch = GetCurrentChallenge();
    if (ch is null) return "\t";
    return ch.Uid + "\t" + ch.Name;
}

// VERIFIED command (Linesight); the path format it accepts is UNVERIFIED.
void ApiLoadMap(const string &in path) { ExecuteCommand("map " + path); }

// ================================================================== 3. state

class LinkInput
{
    bool left = false;
    bool right = false;
    bool accelerate = false;
    bool brake = false;
    bool analog = false;
    int steer = 0;
}

class Snapshot
{
    int raceTime = 0;
    uint8 finished = 0;
    uint8 inRace = 0;
    uint8 paused = 0;
    int cpCount = 0;
    int cpTarget = -1;
    vec3 pos;
    vec3 vel;
    float speedKmh = 0;
    int seq = 0;
}

Net::Socket@ g_listen = null;
Net::Socket@ g_client = null;
bool g_listenFailed = false;
uint64 g_mainMs = 0;

// session (reset when a client connects / disconnects)
bool g_syncMode = true;           // true: game paused between commands; false: free-running
bool g_paused = false;            // we hold the game at SetSpeed(0) in g_pausedState
float g_runSpeed = 1.0f;          // speed used while the game runs (SET_SPEED)
SimulationState@ g_pausedState = null;
int g_pausedRaceTime = 0;
LinkInput g_in;                   // held input, applied every OnRunStep
bool g_steerWasAnalog = false;
int g_op = OP_NONE;
int g_opReq = 0;
bool g_opIsLoad = false;          // OP_LOAD_WAIT reply kind: true = ACK(map info), false = STATE
int g_stepTarget = 0;
bool g_loadArmed = false;         // a NEW race (not the old one) has started since the op began
int g_tick = 0;                   // OnRunStep calls since the client connected
int g_guardRewinds = 0;
bool g_frameStream = false;
int g_streamW = 0;
int g_streamH = 0;
int g_streamMaxFps = 0;
uint64 g_lastStreamMs = 0;
bool g_framePending = false;
int g_pendingReq = 0;
int g_pendingW = 0;
int g_pendingH = 0;
int g_pendingSettle = 0;
int g_stateEvery = 0;

// race (kept across sessions: the map may stay loaded)
SimulationState@ g_startState = null;   // state at race time 0 of the loaded map
int g_startRaceTime = 0;                // its race time; 0 unless the first callback missed t=0
int g_lastRaceTime = -1;
uint64 g_lastCallbackMs = 0;
int g_cpCount = 0;
int g_cpTarget = -1;
bool g_finished = false;
int g_finishTime = 0;
int g_finishTick = 0;                   // g_tick when the finish was first seen in OnRunStep

// payload reader state of the message being handled
uint g_rem = 0;
bool g_bad = false;

// ================================================================== 4. socket I/O

void ResetSession()
{
    g_syncMode = true;
    g_paused = false;
    g_runSpeed = 1.0f;
    @g_pausedState = null;
    g_op = OP_NONE;
    g_framePending = false;
    g_frameStream = false;
    g_stateEvery = 0;
    LinkInput neutral;
    g_in = neutral;
    g_steerWasAnalog = false;
    g_tick = 0;
    g_guardRewinds = 0;
}

// Never leave the game frozen: unpause at normal speed whenever a client goes away.
void DropClient(const string &in reason)
{
    if (g_client !is null) log("TMAgentLink: client dropped (" + reason + ")");
    @g_client = null;
    ResetSession();
    ApiSetSpeed(g_sim, 1.0f);
}

int RdI32()
{
    if (g_rem < 4) { g_bad = true; return 0; }
    g_rem -= 4;
    return g_client.ReadInt32();
}

uint8 RdU8()
{
    if (g_rem < 1) { g_bad = true; return 0; }
    g_rem -= 1;
    return g_client.ReadUint8();
}

float RdF32()
{
    if (g_rem < 4) { g_bad = true; return 0; }
    g_rem -= 4;
    return g_client.ReadFloat();
}

string RdRest()
{
    string s = "";
    if (g_rem > 0) { s = g_client.ReadString(g_rem); g_rem = 0; }
    return s;
}

void RdInput(LinkInput@ inp)
{
    inp.left = RdU8() != 0;
    inp.right = RdU8() != 0;
    inp.accelerate = RdU8() != 0;
    inp.brake = RdU8() != 0;
    inp.analog = RdU8() != 0;
    RdU8(); RdU8(); RdU8();            // padding
    inp.steer = RdI32();
}

bool WaitForBytes(uint n)
{
    uint64 start = Time::Now;
    while (g_client.Available < n) {
        if (Time::Now - start > PAYLOAD_TIMEOUT_MS) return false;
    }
    return true;
}

bool SendHeader(int type, int payloadLen)
{
    return g_client !is null && g_client.Write(type) && g_client.Write(payloadLen);
}

void SendDone(bool ok)
{
    if (!ok) DropClient("write failed");
}

void ReplyAck(int req, const string &in text)
{
    if (g_client is null) return;
    bool ok = SendHeader(REP_ACK, 4 + int(text.Length)) && g_client.Write(req);
    if (ok && text.Length > 0) ok = g_client.Write(text);
    SendDone(ok);
}

void ReplyError(int req, const string &in text)
{
    if (g_client is null) return;
    log("TMAgentLink: error reply: " + text, Severity::Warning);
    bool ok = SendHeader(REP_ERROR, 4 + int(text.Length)) && g_client.Write(req);
    if (ok && text.Length > 0) ok = g_client.Write(text);
    SendDone(ok);
}

void ReplyHello(int req)
{
    string build = PLUGIN_NAME + " " + PLUGIN_VERSION + " protocol " + PROTOCOL_VERSION;
    bool ok = SendHeader(REP_HELLO, 8 + int(build.Length)) && g_client.Write(req)
              && g_client.Write(PROTOCOL_VERSION) && g_client.Write(build);
    SendDone(ok);
}

// ================================================================== 5. snapshots, state, frames

int ReportedRaceTime()
{
    if (g_paused) return g_pausedRaceTime;
    if (g_finished) return g_finishTime;
    return ApiRaceTime(g_sim);
}

Snapshot@ MakeSnapshot(SimulationState@ saved, int raceTime)
{
    Snapshot@ s = Snapshot();
    vec3 pos;
    vec3 vel;
    if (saved !is null) ApiSavedPose(saved, pos, vel);
    else if (ApiInRace(g_sim)) ApiLivePose(g_sim, pos, vel);   // zeros outside a race
    s.raceTime = raceTime;
    s.finished = g_finished ? 1 : 0;
    s.inRace = ApiInRace(g_sim) ? 1 : 0;
    s.paused = g_paused ? 1 : 0;
    s.cpCount = g_cpCount;
    s.cpTarget = g_cpTarget;
    s.pos = pos;
    s.vel = vel;
    s.speedKmh = vel.Length() * 3.6f;
    s.seq = g_tick;
    return s;
}

// State as the client should see it now: from the held snapshot while paused.
Snapshot@ CurrentSnapshot()
{
    if (g_paused && g_pausedState !is null) return MakeSnapshot(g_pausedState, g_pausedRaceTime);
    return MakeSnapshot(null, ReportedRaceTime());
}

bool WriteSnapshot(Snapshot@ s)
{
    return g_client.Write(s.raceTime) && g_client.Write(s.finished) && g_client.Write(s.inRace)
        && g_client.Write(s.paused) && g_client.Write(uint8(0))
        && g_client.Write(s.cpCount) && g_client.Write(s.cpTarget)
        && g_client.Write(s.pos.x) && g_client.Write(s.pos.y) && g_client.Write(s.pos.z)
        && g_client.Write(s.vel.x) && g_client.Write(s.vel.y) && g_client.Write(s.vel.z)
        && g_client.Write(s.speedKmh) && g_client.Write(s.seq);
}

void ReplyState(int req, Snapshot@ s)
{
    if (g_client is null) return;
    SendDone(SendHeader(REP_STATE, 4 + 48) && g_client.Write(req) && WriteSnapshot(s));
}

void PushState()
{
    if (g_client is null) return;
    SendDone(SendHeader(REP_PUSH_STATE, 48) && WriteSnapshot(CurrentSnapshot()));
}

// Capture inside Render() and send as FRAME (req >= 0 -> reply) or PUSH_FRAME (req < 0).
void SendFrame(int req, int w, int h)
{
    if (g_client is null) return;
    vec2 size(w, h);
    array<uint8>@ img = Graphics::CaptureScreenshot(size);
    if (img is null || img.Length != uint(w * h * 4)) {
        if (req >= 0) ReplyError(req, "screenshot failed or has an unexpected size");
        return;
    }
    bool push = req < 0;
    int len = (push ? 0 : 4) + 20 + int(img.Length);
    bool ok = SendHeader(push ? REP_PUSH_FRAME : REP_FRAME, len);
    if (ok && !push) ok = g_client.Write(req);
    ok = ok && g_client.Write(ReportedRaceTime()) && g_client.Write(w) && g_client.Write(h)
            && g_client.Write(PIXEL_BGRA8) && g_client.Write(g_tick) && g_client.Write(img);
    SendDone(ok);
}

// Called from Render(): pending REQUEST_FRAME and the realtime frame stream.
void ServiceFrames()
{
    if (g_client is null) return;
    if (g_framePending) {
        if (g_pendingSettle > 0) {
            g_pendingSettle--;
        } else {
            g_framePending = false;
            SendFrame(g_pendingReq, g_pendingW, g_pendingH);
        }
    }
    if (g_client !is null && g_frameStream && g_streamMaxFps > 0
        && Time::Now - g_lastStreamMs >= uint64(1000 / g_streamMaxFps)) {
        g_lastStreamMs = Time::Now;
        SendFrame(-1, g_streamW, g_streamH);
    }
}

// ================================================================== 6. input, pause, commands

void ApplyInput(SimulationManager@ sm)
{
    if (!ApiInRace(sm)) return;
    sm.SetInputState(InputType::Up, g_in.accelerate ? 1 : 0);
    sm.SetInputState(InputType::Down, g_in.brake ? 1 : 0);
    if (g_in.analog) {
        sm.SetInputState(InputType::Left, 0);
        sm.SetInputState(InputType::Right, 0);
        ApiSetSteer(sm, g_in.steer);
        g_steerWasAnalog = true;
    } else {
        sm.SetInputState(InputType::Left, g_in.left ? 1 : 0);
        sm.SetInputState(InputType::Right, g_in.right ? 1 : 0);
        if (g_steerWasAnalog) { ApiSetSteer(sm, 0); g_steerWasAnalog = false; }
    }
}

// Hold the game at the current tick: save the state, speed 0.
void EnterPause(SimulationManager@ sm, int raceTime)
{
    @g_pausedState = ApiSaveState(sm);
    g_pausedRaceTime = raceTime;
    g_paused = true;
    ApiSetSpeed(sm, 0.0f);
}

void ResumeFromPause()
{
    g_paused = false;
    ApplyInput(g_sim);            // input is set before the first tick after the resume
    ApiSetSpeed(g_sim, g_runSpeed);
}

// The client serializes its requests, so a new LOAD_MAP/RESTART/STEP/SET_MODE while an
// operation is pending means the client gave up on it (timeout): supersede it.
void CancelPendingOp()
{
    if (g_op != OP_NONE) log("TMAgentLink: superseding unfinished operation " + g_op, Severity::Warning);
    g_op = OP_NONE;
}

void CmdSetMode(int req)
{
    int mode = RdI32();
    CancelPendingOp();
    g_frameStream = false;
    g_stateEvery = 0;
    if (mode == 0) {                                   // sync
        g_syncMode = true;
        bool recentTicks = g_lastRaceTime >= 0 && Time::Now - g_lastCallbackMs < 500;
        if (!g_paused && recentTicks) {                // pause at the next OnRunStep
            g_op = OP_PAUSE;
            g_opReq = req;
            return;
        }
    } else {                                           // realtime
        g_syncMode = false;
        if (g_paused) ResumeFromPause();
    }
    ReplyAck(req, "");
}

void CmdLoadMap(int req)
{
    string path = RdRest();
    CancelPendingOp();
    if (path.Length == 0) { ReplyError(req, "LOAD_MAP: empty path"); return; }
    g_paused = false;
    @g_pausedState = null;
    @g_startState = null;
    g_cpCount = 0;
    g_cpTarget = -1;
    g_finished = false;
    g_lastRaceTime = -1;
    g_loadArmed = false;
    g_lastCallbackMs = Time::Now;                      // start of the "no callbacks = loading" gap
    g_op = OP_LOAD_WAIT;
    g_opReq = req;
    g_opIsLoad = true;
    ApiSetSpeed(g_sim, g_runSpeed);                    // the countdown and callbacks need a running game
    ApiLoadMap(path);
}

void CmdRestart(int req)
{
    int method = RdI32();
    CancelPendingOp();
    if (!ApiInRace(g_sim) && g_startState is null) { ReplyError(req, "RESTART: not in a race"); return; }
    LinkInput neutral;
    g_in = neutral;                                    // RESTART clears the held input
    g_opReq = req;
    if (method == RESTART_REWIND && g_startState !is null) {
        g_op = OP_RESTART;                             // done in OnRunStep (rewind needs a running tick)
    } else {
        g_op = OP_LOAD_WAIT;                           // the game's own restart, then wait for t=0
        g_opIsLoad = false;
        g_loadArmed = false;
        g_lastCallbackMs = Time::Now;
        g_finished = false;
        g_cpCount = 0;
        ApiGiveUp(g_sim);
    }
    if (g_paused) ResumeFromPause();
}

void CmdStep(int req)
{
    int n = RdI32();
    LinkInput inp;
    RdInput(inp);
    if (g_bad) return;
    CancelPendingOp();
    if (!g_syncMode) { ReplyError(req, "STEP requires sync mode"); return; }
    if (!g_paused || g_pausedState is null) { ReplyError(req, "STEP: no race is paused (LOAD_MAP / RESTART first)"); return; }
    if (n <= 0 || g_finished) { ReplyState(req, CurrentSnapshot()); return; }
    g_in = inp;
    g_stepTarget = g_pausedRaceTime + n * TICK_MS;
    g_op = OP_STEP;
    g_opReq = req;
    ResumeFromPause();
}

void CmdRequestFrame(int req)
{
    int w = RdI32();
    int h = RdI32();
    int settle = RdI32();
    if (g_bad) return;
    if (g_framePending) { ReplyError(req, "a frame request is already pending"); return; }
    if (w <= 0 || h <= 0 || w > 4096 || h > 4096) { ReplyError(req, "bad frame size"); return; }
    g_framePending = true;
    g_pendingReq = req;
    g_pendingW = w;
    g_pendingH = h;
    g_pendingSettle = settle < 0 ? 0 : settle;
}

void HandleCommand(int type)
{
    int req = RdI32();
    if (type == CMD_HELLO) {
        int ver = RdI32();
        RdRest();
        if (ver != PROTOCOL_VERSION)
            log("TMAgentLink: client speaks protocol " + ver + ", plugin " + PROTOCOL_VERSION, Severity::Warning);
        ReplyHello(req);
    } else if (type == CMD_SET_MODE) {
        CmdSetMode(req);
    } else if (type == CMD_LOAD_MAP) {
        CmdLoadMap(req);
    } else if (type == CMD_RESTART) {
        CmdRestart(req);
    } else if (type == CMD_STEP) {
        CmdStep(req);
    } else if (type == CMD_SET_INPUT) {                // no reply, ever
        RdInput(g_in);
    } else if (type == CMD_REQUEST_FRAME) {
        CmdRequestFrame(req);
    } else if (type == CMD_STREAM_FRAMES) {
        int on = RdI32();
        g_streamW = RdI32();
        g_streamH = RdI32();
        g_streamMaxFps = RdI32();
        g_frameStream = on != 0 && g_streamW > 0 && g_streamH > 0 && g_streamMaxFps > 0 && !g_syncMode;
        ReplyAck(req, "");
    } else if (type == CMD_SET_SPEED) {
        float speed = RdF32();
        if (g_bad) return;
        g_runSpeed = speed;
        if (!g_paused) ApiSetSpeed(g_sim, speed);
        ReplyAck(req, "");
    } else if (type == CMD_EXECUTE) {
        string cmd = RdRest();
        ExecuteCommand(cmd);
        ReplyAck(req, "");
    } else if (type == CMD_GET_STATE) {
        ReplyState(req, CurrentSnapshot());
    } else if (type == CMD_CLOSE) {
        ReplyAck(req, "");
        DropClient("client closed");
    } else if (type == CMD_STREAM_STATE) {
        g_stateEvery = RdI32();
        if (g_syncMode) g_stateEvery = 0;
        ReplyAck(req, "");
    } else if (type == CMD_PING) {                     // ACK text = diagnostics for the client
        ReplyAck(req, "guard_rewinds=" + g_guardRewinds + " tick=" + g_tick + " op=" + g_op);
    } else {
        log("TMAgentLink: unknown command type " + type, Severity::Warning);
        ReplyError(req, "unknown command type " + type);
    }
}

// Non-blocking: handle every complete message that has arrived (bounded per call).
void PollSocket()
{
    for (int i = 0; i < MAX_MESSAGES_PER_POLL; i++) {
        if (g_client is null) return;
        if (g_client.Available < HEADER_SIZE) return;
        int type = g_client.ReadInt32();
        int len = g_client.ReadInt32();
        if (len < 0 || uint(len) > MAX_COMMAND_PAYLOAD) { DropClient("bad payload length " + len); return; }
        if (!WaitForBytes(uint(len))) { DropClient("timeout waiting for a payload"); return; }
        g_rem = uint(len);
        g_bad = false;
        HandleCommand(type);
        if (g_client is null) return;
        if (g_rem > 0) { g_client.ReadBytes(g_rem); g_rem = 0; }     // skip unread payload
        if (g_bad) log("TMAgentLink: short payload for command " + type, Severity::Warning);
    }
}

// ================================================================== 7. race callbacks

// Finish an OP_LOAD_WAIT: the new race reached time 0.
void CompleteStartWait(SimulationManager@ sm, int rt)
{
    @g_startState = ApiSaveState(sm);
    g_startRaceTime = rt;                              // normally 0; the client checks and reports otherwise
    g_cpCount = 0;
    g_finished = false;
    g_finishTime = 0;
    g_op = OP_NONE;
    if (g_syncMode) EnterPause(sm, rt);
    if (g_opIsLoad) ReplyAck(g_opReq, ApiMapInfo());
    else ReplyState(g_opReq, CurrentSnapshot());
}

// Rewind to the saved race-time-0 state (RESTART, method REWIND).
void CompleteRewind(SimulationManager@ sm)
{
    ApiRewind(sm, g_startState);
    g_cpCount = 0;
    g_finished = false;
    g_finishTime = 0;
    g_lastRaceTime = g_startRaceTime;
    g_op = OP_NONE;
    if (g_syncMode) {
        @g_pausedState = g_startState;                 // report the saved state, not a live read
        g_pausedRaceTime = g_startRaceTime;
        g_paused = true;
        ApiSetSpeed(sm, 0.0f);
    }
    ReplyState(g_opReq, MakeSnapshot(g_startState, g_startRaceTime));
}

void AdvanceOp(SimulationManager@ sm, int rt)
{
    if (g_op == OP_LOAD_WAIT) {
        if (g_loadArmed && rt >= 0) CompleteStartWait(sm, rt);
    } else if (g_op == OP_RESTART) {
        CompleteRewind(sm);
    } else if (g_op == OP_STEP) {
        if (!g_finished && ApiPlayerFinished(sm)) { g_finished = true; g_finishTime = rt; g_finishTick = g_tick; }
        // A finish normally completes the STEP in OnCheckpointCountChanged, which TMI calls right
        // after this callback and which carries the final checkpoint count; one tick later is
        // the fallback if that callback never comes.
        bool done = g_finished ? g_tick > g_finishTick : rt >= g_stepTarget;
        if (done) {
            g_op = OP_NONE;
            EnterPause(sm, g_finished ? g_finishTime : rt);
            ReplyState(g_opReq, CurrentSnapshot());
        }
    } else if (g_op == OP_PAUSE) {
        g_op = OP_NONE;
        EnterPause(sm, rt);
        ReplyAck(g_opReq, "");
    }
}

void OnRunStep(SimulationManager@ sm)
{
    if (g_client is null) return;

    // Paused guard: no tick should run while paused. If SetSpeed(0) did not stop the
    // engine in time, put the simulation back to the held state (UNVERIFIED need).
    if (g_paused) {
        if (g_pausedState !is null) ApiRewind(sm, g_pausedState);
        ApiSetSpeed(sm, 0.0f);
        g_guardRewinds++;
        if (g_guardRewinds <= 3) log("TMAgentLink: paused guard rewound an extra tick");
        return;
    }

    int rt = ApiRaceTime(sm);
    g_tick++;
    uint64 now = Time::Now;
    // A new race started if we saw the countdown (rt < 0), time went backwards, or the
    // callbacks paused for a while (map loading). Used only while OP_LOAD_WAIT is pending.
    if (rt < 0 || rt < g_lastRaceTime || now - g_lastCallbackMs > LOAD_GAP_MS) g_loadArmed = true;
    g_lastCallbackMs = now;
    g_lastRaceTime = rt;

    PollSocket();
    if (g_client is null || g_paused) return;
    AdvanceOp(sm, rt);
    if (g_client is null || g_paused) return;

    if (rt >= 0) ApplyInput(sm);
    if (g_stateEvery > 0 && (g_tick % g_stateEvery) == 0) PushState();
}

void OnCheckpointCountChanged(SimulationManager@ sm, int current, int target)
{
    g_cpCount = current;
    g_cpTarget = target;
    if (current != target || g_client is null) return;
    // Finish line: current == target (Linesight uses the same rule).
    g_finished = true;
    g_finishTime = ApiRaceTime(sm);
    ApiPreventFinish(sm);                               // keep the simulation alive after the finish
    if (g_client !is null && g_op == OP_STEP && !g_paused) {
        g_op = OP_NONE;                                 // finish ends a STEP early
        EnterPause(sm, g_finishTime);
        ReplyState(g_opReq, CurrentSnapshot());
    }
}

void OnGameStateChanged(TM::GameState state)
{
    if (g_op == OP_LOAD_WAIT) g_loadArmed = true;       // menus/loading seen: the old race is gone
}

// ================================================================== 8. entry points

void StartListening()
{
    if (g_listen !is null) return;
    int port = int(GetVariableDouble("tmagent_port"));
    if (port <= 0) port = DEFAULT_PORT;
    @g_listen = Net::Socket();
    if (!g_listen.Listen(HOST, uint16(port))) {
        log("TMAgentLink: cannot listen on " + HOST + ":" + port + " (port in use?)", Severity::Error);
        @g_listen = null;
        g_listenFailed = true;
        return;
    }
    log("TMAgentLink listening on " + HOST + ":" + port);
}

// Deferred start: `set tmagent_port` from the launch command line is applied only after
// the command queue ran, so Main() queues this command (same approach as Linesight).
void OnListenCommand(int fromTime, int toTime, const string &in commandLine, const array<string> &in args)
{
    StartListening();
}

void OnStatusCommand(int fromTime, int toTime, const string &in commandLine, const array<string> &in args)
{
    log("TMAgentLink: listening=" + (g_listen !is null) + " client=" + (g_client !is null)
        + " sync=" + g_syncMode + " paused=" + g_paused + " op=" + g_op + " tick=" + g_tick
        + " raceTime=" + g_lastRaceTime + " cp=" + g_cpCount + "/" + g_cpTarget
        + " finished=" + g_finished + " guardRewinds=" + g_guardRewinds);
}

void AcceptClient()
{
    if (g_listen is null) return;
    Net::Socket@ s = g_listen.Accept(0);
    if (s is null) return;
    if (g_client !is null) DropClient("replaced by a new connection");
    @g_client = s;
    s.NoDelay = true;
    ResetSession();
    log("TMAgentLink: client connected from " + s.RemoteIP);
}

// Frame boundary while paused: no tick is running now. If the simulation is not at the held
// state (a tick slipped through after SetSpeed(0) and OnRunStep did not run after it),
// put it back. UNVERIFIED need; calls RewindToState from Render().
void CheckPausedDrift()
{
    if (!g_paused || g_pausedState is null) return;
    if (ApiRaceTime(g_sim) == g_pausedRaceTime) return;
    ApiRewind(g_sim, g_pausedState);
    g_guardRewinds++;
    if (g_guardRewinds <= 3) log("TMAgentLink: paused drift corrected in Render (race time " + ApiRaceTime(g_sim) + " vs " + g_pausedRaceTime + ")");
}

void Render()
{
    if (g_listen is null && !g_listenFailed && Time::Now - g_mainMs > LISTEN_FALLBACK_MS) StartListening();
    AcceptClient();
    if (g_client is null) return;
    PollSocket();
    CheckPausedDrift();
    ServiceFrames();
}

void Main()
{
    g_mainMs = Time::Now;
    RegisterVariable("tmagent_port", 8477.0);
    RegisterCustomCommand("tmagent_listen", "Start the TMAgentLink TCP server", OnListenCommand);
    RegisterCustomCommand("tmagent_status", "Print the TMAgentLink session state", OnStatusCommand);
    CommandList cmds;
    cmds.Content = "tmagent_listen";
    cmds.Process();
}

void OnDisabled()
{
    DropClient("plugin disabled");
    @g_listen = null;
}

PluginInfo@ GetPluginInfo()
{
    PluginInfo info;
    info.Name = PLUGIN_NAME;
    info.Author = "tmagent";
    info.Version = PLUGIN_VERSION;
    info.Description = "TCP bridge for the tmagent Python client (tick-exact stepping, frame capture).";
    return info;
}
