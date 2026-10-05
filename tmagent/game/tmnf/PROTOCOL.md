# TMAgentLink protocol (version 1)

Our own binary protocol between the TMInterface plugin
(`plugin/TMAgentLink.as`, TCP **server** on `127.0.0.1:game.tmi_port`, default 8477)
and the Python client (`client.py`). Code: `protocol.py` (Python side),
`plugin/TMAgentLink.as` (hand-written mirror), `fake_server.py` (Python fake of the
plugin, used by tests). Any layout change bumps `PROTOCOL_VERSION` in both.

Facts about the game API and their sources are in the header of
`TMAgentLink.as` and in `docs/research.md`; anything not confirmed on a real install
is marked UNVERIFIED there and in the Python constants.

## Framing

All numbers little-endian. Every message:

```
int32 type | int32 payload_length | payload
```

Every **command** payload starts with `int32 req_id` (chosen by the client, > 0; 0
for `SET_INPUT`). Every **reply** to a command starts with the same `req_id`, so a late
reply to a request that already timed out is recognized and dropped. **Pushed**
messages (`PUSH_STATE`, `PUSH_FRAME`) have no req_id. Strings are raw UTF-8 bytes
filling the rest of the payload (no length prefix, no terminator).

Limits: the plugin accepts command payloads up to 1 MiB and drops the client on a
bad length. Replies are up to 64 MiB.

## Shared payloads

`INPUT` (12 bytes): `u8 left, u8 right, u8 accelerate, u8 brake, u8 analog, 3 pad
bytes, i32 steer`. `steer` in [-65536, 65536] (TMI convention, negative = left) is used
only when `analog != 0` (then left/right are ignored). `accelerate` = TMI `Up`, `brake` =
TMI `Down`.

`STATE` (48 bytes): `i32 race_time_ms, u8 finished, u8 in_race, u8 paused, u8 pad,
i32 cp_count, i32 cp_target, f32 pos[3], f32 vel[3], f32 speed_kmh, i32 seq`.
`cp_target` is -1 until the first checkpoint event (including the finish) of the
map was seen; `seq` counts plugin physics callbacks since the client connected (use it
to measure the tick rate). `speed_kmh` = |velocity| * 3.6, velocity in m/s.

`FRAME` payload after the req_id (and `PUSH_FRAME` payload): `i32 race_time_ms,
i32 width, i32 height, i32 pixel_format (0 = BGRA8), i32 seq`, then `width*height*4`
raw bytes.

## Commands (client -> plugin)

| id | name | payload after req_id | reply |
|----|------|----------------------|-------|
| 1 | HELLO | `i32 client_version`, UTF-8 client name | HELLO (105): `i32 plugin_version`, UTF-8 build string |
| 2 | SET_MODE | `i32 mode` (0 sync, 1 realtime) | ACK |
| 3 | LOAD_MAP | UTF-8 path (as given to the game's `map` command) | ACK with text `uid<TAB>name`, once the race is ready at race time 0 |
| 4 | RESTART | `i32 method` (0 rewind to the saved t=0 state, 1 give up = game restart) | STATE at race time 0 |
| 5 | STEP | `i32 n_ticks`, INPUT | STATE after n ticks (or at the finish) |
| 6 | SET_INPUT | INPUT | none, ever |
| 7 | REQUEST_FRAME | `i32 w, i32 h, i32 settle` | FRAME |
| 8 | STREAM_FRAMES | `i32 on, i32 w, i32 h, i32 max_fps` | ACK; PUSH_FRAME messages (realtime only) |
| 9 | SET_SPEED | `f32 speed` | ACK |
| 10 | EXECUTE | UTF-8 TMI console command | ACK |
| 11 | GET_STATE | none | STATE |
| 12 | CLOSE | none | ACK, then the plugin drops the client |
| 13 | STREAM_STATE | `i32 every_n_ticks` (0 = off) | ACK; PUSH_STATE messages (realtime only) |
| 14 | PING | none | ACK with diagnostics text `guard_rewinds=<n> tick=<n> op=<n>` |

Plugin -> client types: ACK 101 (`UTF-8 text`), ERROR 102 (`UTF-8 text`), STATE 103,
FRAME 104, HELLO 105, PUSH_STATE 106, PUSH_FRAME 107. Commands without a documented
reply (`SET_INPUT`) never get any reply, not even ERROR, so reply order stays
trivial; errors for them are only logged in the game console.

### Handshake and versions

The client sends HELLO first. The plugin always answers with its own protocol version
and build string; the client raises `ProtocolMismatch` if the versions differ. A new
connection **replaces** the previous one (newest wins), so a crashed client can simply
reconnect. When a client goes away (CLOSE, replaced, write failure, plugin disabled)
the plugin resets its session and sets the game speed back to 1.0, it never leaves the
game frozen.

### Semantics

**Modes.** A fresh session starts in sync mode. `SET_MODE sync`: pause the game at the
next physics tick (ACK is sent when it is paused; immediately if no race is running),
disable the realtime streams. `SET_MODE realtime`: resume at the speed last given with
`SET_SPEED`.

**Pausing.** "Paused" means `SetSpeed(0)` with a saved `SimulationState`; the plugin
thread is not blocked, `Render()` keeps being called. STATE replies while paused are
built from the saved state, so they do not depend on what the engine does afterwards.
Two guards keep the held state exact even if `SetSpeed(0)` takes a tick to act
(UNVERIFIED whether they are ever needed): `OnRunStep` while paused rewinds to the held
state, and `Render()` rewinds if the race time differs from the held one. The PING
diagnostics (`guard_rewinds`) count these rewinds; `tools/tmnf_smoke.py` reports them and
checks that n single-tick STEPs end where one n-tick STEP ends. Known limitation: if the
engine reads a stale `RaceTime` right after `RewindToState` and also overshoots after
`SetSpeed(0)`, a guard rewind can make the next STEP lose a tick; a lower `render_speed`
(fewer ticks per rendered frame) reduces the exposure.

**LOAD_MAP** runs `map <path>`, un-pauses (the countdown must run), waits until a
*new* race reaches race time >= 0 (detected by a negative race time during the
countdown, a race time going backwards, a game-state change, or a >= 1 s gap in
`OnRunStep`), saves the state at race time 0 (used by RESTART), pauses in sync mode and
sends ACK with the map's uid and name. Allow a long client timeout (default 120 s).

**RESTART** method 0 rewinds to the saved t=0 state at the next physics callback (the
game runs for a moment even when paused), clears the held input and checkpoint
counters, pauses again in sync mode. Method 1 (or no saved state) calls `GiveUp()` and
waits for the new race like LOAD_MAP.

**STEP(n, input)** (sync mode, game paused): set the held input, resume at the speed
from `SET_SPEED`, and pause again at the first callback whose race time is >= the
paused race time + 10 * n, then reply STATE. The target is race-time based, so it is
exact whether TMI calls `OnRunStep` before or after each tick. The input is set before
the first tick, so it acts on the n ticks that follow the paused state. A finish ends
the STEP early with `finished = 1`; STEP on a finished race returns the state at once;
`n <= 0` returns the state without running. STEP in realtime mode or without a paused
race is an ERROR. Frame `t` (REQUEST_FRAME after the step) shows the paused state at
race time `t`, before the next input is applied (same convention as `interfaces.py`).

**SET_INPUT** (realtime): replaces the held input; the plugin applies it with
`SetInputState` at every physics callback, so it acts from the next tick on.

**REQUEST_FRAME** is captured in `Render()` with `Graphics::CaptureScreenshot(vec2(w, h))`.
`settle` is the number of `Render()` calls to wait after the request arrived before
capturing (0 = capture in the same call; the default 1 guards against a screenshot of
the previous scene render, UNVERIFIED need). The reply carries the race time of the
held state. The pixel row order is assumed top-down (UNVERIFIED, `client.CAPTURE_FLIP_VERTICAL`).

**STREAM_FRAMES / STREAM_STATE** (realtime): `PUSH_FRAME` from `Render()` at most
`max_fps` times per second, `PUSH_STATE` from the physics callback every n ticks. The
Python reader thread only stores the newest message, so consumers never wait.

**Errors.** A command the plugin cannot execute gets an ERROR reply with text. A new
LOAD_MAP / RESTART / STEP / SET_MODE while another is pending supersedes it (the client
serializes requests, so this only happens after a client-side timeout).

## Blocking model (when the plugin waits on the socket) and why

* **Reading is non-blocking.** `OnRunStep` and `Render` poll `Socket.Available`; a
  message is read only when at least its 8-byte header has arrived (at most 32 messages
  per poll). The game thread never waits for the client to send something.
  `Net::Socket.Available` is confirmed by three independent plugins (Linesight bridge,
  the donadigo example, a TMNF telemetry plugin that polls it inside `OnRunStep`),
  which is why no lockstep protocol is needed.
* **Bounded waits.** Once a header has arrived, the plugin waits at most 2 s for the
  rest of that message (the client writes each message with one `sendall`). `Write`
  can block only if the client stops reading (full TCP buffer), the Python reader thread
  always drains the socket.
* **Sync mode** is paused with `SetSpeed(0)`, not by blocking the thread. Commands that
  arrive while paused are noticed in the next `Render()` (<= one rendered frame of
  latency). A blocking loop inside `OnRunStep` would also pause the sim, but it would
  freeze `Render()` and make screenshots of the paused state impossible, which the
  frame capture needs.
* **Realtime mode** never blocks: input and commands are polled each tick, frames and
  states are pushed. The control thread's `set_action` is one small `send` (deduplicated:
  only on change) and never waits for the model or a reply.
* Rejected alternative: lockstep for realtime (plugin sends a TICK and waits for an
  INPUT reply each physics tick). It would stall the game whenever the Python side
  hiccups and adds a round trip per tick. It stays the fallback design if `Available`
  turns out to be unreliable on a given TMI version (then only the plugin changes).

## Client behaviour

`TMAgentClient`: connects with retries until `timeout`, handshakes, runs one reader
thread that demultiplexes replies (by `req_id`) from pushes, serializes requests (one
outstanding), and raises `TMAgentTimeout`, `ConnectionLost`, `PluginError`,
`ProtocolMismatch`. After `ConnectionLost`, call `connect()` again; after a timeout
of LOAD_MAP/RESTART/STEP call `TMNFGame.load_map` / `start_race` again (the plugin state
of the abandoned operation is superseded).
