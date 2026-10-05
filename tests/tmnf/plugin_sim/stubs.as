// Dev-only stubs: TMI 2.x API surface used by TMAgentLink.as + a simulated game engine.
// Declarations follow as.predefined (TMNF folder of sashi0034/angel-lsp) and Linesight.

// ------------------------------------------------------------------ math
class vec2 {
    float x = 0; float y = 0;
    vec2() {}
    vec2(float a, float b) { x = a; y = b; }
}
class vec3 {
    float x = 0; float y = 0; float z = 0;
    vec3() {}
    vec3(float a, float b, float c) { x = a; y = b; z = c; }
    float Length() { return float(sys_sqrt(x * x + y * y + z * z)); }
}
class mat3 { vec3 x; vec3 y; vec3 z; }
class iso4 { mat3 Rotation; vec3 Position; }

enum Severity { Info = 0, Success = 1, Warning = 2, Error = 3 }
void log(const string &in str, Severity severity = Severity::Info) { sys_log(str); }
void print(const string &in str, Severity severity = Severity::Info) { sys_log(str); }

class PluginInfo { string Author; string Name; string Description; string Version; }

namespace Time { uint64 get_Now() property { return sys_now_ms(); } }

// ------------------------------------------------------------------ sockets
namespace Net {
class Socket {
    int fd = -1;
    int lfd = -1;
    Socket() {}
    ~Socket() { if (fd >= 0) sys_close(fd); if (lfd >= 0) sys_close(lfd); }
    bool Connect(const string &in host, uint16 port, uint timeout = 0xFFFFFFFF) { return false; }
    bool Listen(const string &in host, uint16 port) { lfd = sys_listen(host, int(port)); return lfd >= 0; }
    Socket@ Accept(uint timeoutMs = 0xFFFFFFFF) {
        if (lfd < 0) return null;
        int c = sys_accept(lfd);
        if (c < 0) return null;
        Socket s; s.fd = c;
        return s;
    }
    uint get_Available() property { return sys_avail(fd); }
    void set_NoDelay(bool value) property { if (value) sys_nodelay(fd); }
    string get_RemoteIP() property { return "127.0.0.1"; }
    string ReadString(uint bytes) { return sys_read_str(fd, bytes); }
    array<uint8>@ ReadBytes(uint bytes) { return sys_read_bytes(fd, bytes); }
    int8 ReadInt8() { return int8(sys_read_int(fd, 1)); }
    uint8 ReadUint8() { return uint8(sys_read_int(fd, 1)); }
    int16 ReadInt16() { return int16(sys_read_int(fd, 2)); }
    uint16 ReadUint16() { return uint16(sys_read_int(fd, 2)); }
    int ReadInt32() { return int(sys_read_int(fd, 4)); }
    uint ReadUint32() { return uint(sys_read_int(fd, 4)); }
    int64 ReadInt64() { return sys_read_int(fd, 8); }
    uint64 ReadUint64() { return uint64(sys_read_int(fd, 8)); }
    float ReadFloat() { return sys_read_f32(fd); }
    double ReadDouble() { return 0; }
    bool Write(const string &in data) { return sys_write_str(fd, data); }
    bool Write(const array<uint8> &in data) { return sys_write_bytes(fd, data); }
    bool Write(int8 value) { return sys_write_int(fd, value, 1); }
    bool Write(uint8 value) { return sys_write_int(fd, value, 1); }
    bool Write(int16 value) { return sys_write_int(fd, value, 2); }
    bool Write(uint16 value) { return sys_write_int(fd, value, 2); }
    bool Write(int value) { return sys_write_int(fd, value, 4); }
    bool Write(uint value) { return sys_write_int(fd, value, 4); }
    bool Write(int64 value) { return sys_write_int(fd, value, 8); }
    bool Write(uint64 value) { return sys_write_int(fd, int64(value), 8); }
    bool Write(float value) { return sys_write_f32(fd, value); }
    bool Write(double value) { return sys_write_f64(fd, value); }
}
}

// ------------------------------------------------------------------ TM types
enum InputType { None = -1, Down = 0, Up = 1, Left = 2, Right = 3, Steer = 4, Gas = 5, Respawn = 6, GiveUp = 7, Horn = 8, FakeFinish = 9 }
enum CommandListProcessOption { OnlyParse = 0, QueueAndExecute = 1, ExecuteImmediately = 2 }
enum ExecuteCommandFlags { None = 0, AppendHistory = 1, Echo = 2, SuppressOutput = 8, Default = 3 }

namespace TM {
    enum GameState { None = 0, StartUp = 16, Menus = 32, LocalRace = 512 }
    class HmsStateDyna { iso4 Location; vec3 LinearSpeed; }
    class HmsDyna { HmsStateDyna CurrentState; }
    class PlayerInfo { bool RaceFinished = false; uint CurCheckpointCount = 0; }
    class GameCtnChallenge {
        string get_Uid() property { return "FAKEUID123"; }
        string get_Name() property { return "FakeMap"; }
    }
}

funcdef void OnCustomCommand(int fromTime, int toTime, const string &in commandLine, const array<string> &in args);
array<string> g_cmdNames;
array<OnCustomCommand@> g_cmdFuncs;

void RegisterCustomCommand(const string &in name, const string &in description, OnCustomCommand@ callback)
{
    g_cmdNames.insertLast(name);
    g_cmdFuncs.insertLast(callback);
}
bool RegisterVariable(const string &in name, bool defaultVal) { return true; }
bool RegisterVariable(const string &in name, double defaultVal) { return true; }
bool RegisterVariable(const string &in name, const string &in defaultVal) { return true; }
double GetVariableDouble(const string &in name) { return sys_var(name); }

void RunCommandLine(const string &in line)
{
    array<string> args;
    for (uint i = 0; i < g_cmdNames.length(); i++) {
        if (line == g_cmdNames[i]) { g_cmdFuncs[i](0, 0, line, args); return; }
    }
    if (line.length() > 4 && line.substr(0, 4) == "map ") GetEngine().RequestMapLoad(line.substr(4));
}

class CommandList {
    string Content;
    CommandList() {}
    void Process(CommandListProcessOption option = CommandListProcessOption::QueueAndExecute) { RunCommandLine(Content); }
}

void ExecuteCommand(const string &in input, ExecuteCommandFlags flags = ExecuteCommandFlags::Default) { RunCommandLine(input); }
TM::GameState GetCurrentGameState() { return TM::GameState::LocalRace; }
TM::GameCtnChallenge@ GetCurrentChallenge() { return TM::GameCtnChallenge(); }

namespace Graphics {
    array<uint8>@ CaptureScreenshot(vec2 &inout size) {
        int w = int(size.x);
        int h = int(size.y);
        array<uint8> img(w * h * 4);
        int t = GetEngine().renderedRaceTime;
        for (int i = 0; i < w * h; i++) {
            img[i * 4 + 0] = uint8(t & 255);
            img[i * 4 + 1] = uint8((t >> 8) & 255);
            img[i * 4 + 2] = uint8((t >> 16) & 255);
            img[i * 4 + 3] = 255;
        }
        return img;
    }
}

// ------------------------------------------------------------------ simulated engine
class SimulationState {
    float x = 0; float y = 0; float hd = 0; float v = 0;
    int raceTime = 0; int cp = 0; bool finished = false;
    TM::HmsDyna dyna;
    TM::HmsDyna@ get_Dyna() property { return dyna; }
}

class SimulationManager {
    FakeEngine@ e;
    int get_RaceTime() property { return e.lagActive ? e.lagValue : e.raceTime; }
    int get_TickTime() property { return e.raceTime; }
    bool get_InRace() property { return e.phase == 2; }
    TM::PlayerInfo@ get_PlayerInfo() property { return e.player; }
    TM::HmsDyna@ get_Dyna() property { e.RefreshDyna(); return e.dyna; }
    void SetSpeed(float speed) { e.speed = speed; }
    void SetInputState(InputType t, int value) { e.SetInput(t, value); }
    SimulationState@ SaveState() { return e.Save(); }
    void RewindToState(const SimulationState &in st) { e.Restore(st); }
    void GiveUp() { e.GiveUp(); }
    void PreventSimulationFinish() { e.preventFinish = true; }
}

const int PH_MENU = 0;
const int PH_LOADING = 1;
const int PH_RACE = 2;
const int VAR_CB_BEFORE = 1;          // callback precedes the tick instead of following it
const int VAR_SPEED_IMMEDIATE = 2;    // SetSpeed(0) stops the tick loop at once
const int VAR_NO_STATE_EVENTS = 4;    // no OnGameStateChanged
const int VAR_NO_COUNTDOWN_CB = 8;    // no OnRunStep while the countdown runs
const int VAR_REWIND_LAGS = 16;       // RaceTime reads the old value right after RewindToState (until next tick)

class FakeEngine {
    SimulationManager@ sim;
    TM::PlayerInfo player;
    TM::HmsDyna dyna;
    int variant = 0;
    int phase = PH_MENU;
    float speed = 1;
    double accum = 0;
    int raceTime = 0;
    int renderedRaceTime = 0;
    bool finished = false;
    bool preventFinish = false;
    int cp = 0;
    int cpTarget = 3;
    float x = 0; float y = 0; float hd = 0; float v = 0;
    bool left = false; bool right = false; bool up = false; bool down = false;
    int steer = 0;
    bool pendingMapLoad = false;
    uint64 loadStart = 0;
    float finishX = 80;
    int ticks = 0;
    bool lagActive = false;
    int lagValue = 0;
    int cbCount = 0;

    FakeEngine() { @sim = SimulationManager(); @sim.e = this; }

    void RefreshDyna() {
        dyna.CurrentState.Location.Position.x = x;
        dyna.CurrentState.Location.Position.y = 0;
        dyna.CurrentState.Location.Position.z = y;
        dyna.CurrentState.LinearSpeed.x = float(v * sys_cos(hd));
        dyna.CurrentState.LinearSpeed.y = 0;
        dyna.CurrentState.LinearSpeed.z = float(v * sys_sin(hd));
    }
    void RequestMapLoad(const string &in path) { pendingMapLoad = true; sys_log("[engine] map command: " + path); }
    void ResetCar() { lagActive = false; x = 0; y = 0; hd = 0; v = 0; finished = false; cp = 0; player.RaceFinished = false; player.CurCheckpointCount = 0; left = right = up = down = false; steer = 0; }
    void GiveUp() { raceTime = -2000; ResetCar(); phase = PH_RACE; }
    void SetInput(InputType t, int value) {
        if (phase != PH_RACE) return;
        if (t == InputType::Left) left = value != 0;
        else if (t == InputType::Right) right = value != 0;
        else if (t == InputType::Up) up = value != 0;
        else if (t == InputType::Down) down = value != 0;
        else if (t == InputType::Steer) steer = value;
    }
    SimulationState@ Save() {
        SimulationState s;
        s.x = x; s.y = y; s.hd = hd; s.v = v; s.raceTime = raceTime; s.cp = cp; s.finished = finished;
        s.dyna.CurrentState.Location.Position.x = x;
        s.dyna.CurrentState.Location.Position.z = y;
        s.dyna.CurrentState.LinearSpeed.x = float(v * sys_cos(hd));
        s.dyna.CurrentState.LinearSpeed.z = float(v * sys_sin(hd));
        return s;
    }
    void Restore(const SimulationState &in s) {
        if ((variant & VAR_REWIND_LAGS) != 0) { lagActive = true; lagValue = raceTime; }
        x = s.x; y = s.y; hd = s.hd; v = s.v; raceTime = s.raceTime; cp = s.cp; finished = s.finished;
        player.RaceFinished = finished; player.CurCheckpointCount = cp;
    }
    void Physics() {
        lagActive = false;
        if (raceTime < 0) { raceTime += 10; return; }
        if (finished) return;
        if (up) v += 0.2f;
        if (down) { v -= 0.4f; if (v < 0) v = 0; }
        v *= 0.999f;
        float st = (right ? 1.0f : 0.0f) - (left ? 1.0f : 0.0f) + steer / 65536.0f;
        hd += st * 0.015f;
        x += float(v * sys_cos(hd) * 0.01);
        y += float(v * sys_sin(hd) * 0.01);
        raceTime += 10;
    }
    bool CheckCheckpoints() {   // returns true if a cp callback is due
        if (finished || raceTime < 0) return false;
        int newCp = cp;
        if (cp == 0 && x >= 25) newCp = 1;
        if (cp == 1 && x >= 55) newCp = 2;
        if (x >= finishX) newCp = cpTarget;
        if (newCp != cp) {
            cp = newCp; player.CurCheckpointCount = cp;
            if (cp == cpTarget) { finished = true; player.RaceFinished = true; }
            return true;
        }
        return false;
    }
    void Tick() {
        ticks++;
        bool seeCb = !(raceTime < 0 && (variant & VAR_NO_COUNTDOWN_CB) != 0);
        if ((variant & VAR_CB_BEFORE) != 0) {
            if (seeCb) OnRunStep(sim);
            if (phase != PH_RACE) return;
        }
        Physics();
        bool cpDue = CheckCheckpoints();
        if ((variant & VAR_CB_BEFORE) == 0 && seeCb) OnRunStep(sim);
        if (cpDue) {
            OnCheckpointCountChanged(sim, cp, cpTarget);
            if (finished && !preventFinish) { phase = PH_MENU; sys_log("[engine] race ended by finish (PreventSimulationFinish not called)"); }
        }
    }
    void Frame(int frameMs) {
        if (pendingMapLoad) {
            pendingMapLoad = false;
            phase = PH_LOADING; loadStart = sys_now_ms(); accum = 0;
            if ((variant & VAR_NO_STATE_EVENTS) == 0) OnGameStateChanged(TM::GameState::Menus);
        }
        if (phase == PH_LOADING && sys_now_ms() - loadStart > 300) {
            phase = PH_RACE; raceTime = -2000; ResetCar(); preventFinish = false;
            if ((variant & VAR_NO_STATE_EVENTS) == 0) OnGameStateChanged(TM::GameState::LocalRace);
        }
        if (phase == PH_RACE) {
            accum += frameMs * speed;
            while (accum >= 10 && phase == PH_RACE) {
                if ((variant & VAR_SPEED_IMMEDIATE) != 0 && speed <= 0) break;
                accum -= 10;
                Tick();
            }
            if (speed <= 0) accum = 0;
        }
        renderedRaceTime = raceTime;
        Render();
    }
}

FakeEngine@ g_engine = null;
FakeEngine@ GetEngine() { if (g_engine is null) @g_engine = FakeEngine(); return g_engine; }
SimulationManager@ GetSimulationManager() { return GetEngine().sim; }

const int FRAME_MS = 8;
void HostInit() { GetEngine().variant = int(sys_var("variant")); }
void HostFrame() { sys_sleep_ms(FRAME_MS); GetEngine().Frame(FRAME_MS); }
