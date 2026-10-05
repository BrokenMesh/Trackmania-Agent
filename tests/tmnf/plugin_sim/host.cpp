// Dev-only harness for plugin/TMAgentLink.as (used by tests/tmnf/test_plugin_sim.py).
//
// Compiles stubs.as + the plugin with AngelScript 2.35 and runs the simulated TMI engine
// (HostFrame in stubs.as) so the real Python client can drive the plugin over TCP.
// Build (Debian/Ubuntu: apt install angelscript-dev libangelscript-addon2.35.1t64):
//   g++ -std=c++17 -DAS_USE_NAMESPACE host.cpp -I/usr/include/angelscript \
//       /usr/lib/x86_64-linux-gnu/libangelscript-addon.so.2.35.1 -langelscript -o host
// Run: host stubs.as TMAgentLink.as <port> <run_seconds> [variant=<bitmask>]
// Variant bits: see VAR_* in stubs.as (tick/callback order, SetSpeed latency, missing events).
#include <angelscript.h>
#include <scriptstdstring.h>
#include <scriptarray.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>

using std::string;
using namespace AngelScript;
static asIScriptEngine* engine;
static std::map<string, double> g_vars;
static auto t0 = std::chrono::steady_clock::now();

static void MessageCallback(const asSMessageInfo* msg, void*) {
  const char* t = msg->type == asMSGTYPE_ERROR ? "ERROR" : msg->type == asMSGTYPE_WARNING ? "WARN" : "INFO";
  printf("%s (%d,%d) %s : %s\n", msg->section, msg->row, msg->col, t, msg->message);
  fflush(stdout);
}
static unsigned long long sys_now_ms() {
  return (unsigned long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() + 100000ULL;
}
static void sys_log(const string& s) { printf("[plugin] %s\n", s.c_str()); fflush(stdout); }
static int sys_listen(const string& host, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, host.c_str(), &a.sin_addr);
  if (bind(fd, (sockaddr*)&a, sizeof a) < 0 || listen(fd, 4) < 0) { close(fd); return -1; }
  fcntl(fd, F_SETFL, O_NONBLOCK);
  return fd;
}
static int sys_accept(int lfd) {
  int c = accept(lfd, nullptr, nullptr);
  if (c < 0) return -1;
  return c;
}
static unsigned sys_avail(int fd) { int n = 0; if (ioctl(fd, FIONREAD, &n) < 0) return 0; return n < 0 ? 0 : n; }
static void sys_nodelay(int fd) { int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); }
static void sys_close(int fd) { if (fd >= 0) close(fd); }
static bool readn(int fd, void* buf, size_t n) {
  size_t got = 0; while (got < n) { ssize_t r = recv(fd, (char*)buf + got, n - got, 0); if (r <= 0) return false; got += r; } return true;
}
static long long sys_read_int(int fd, int nbytes) {
  unsigned char b[8] = {0}; readn(fd, b, nbytes);
  if (nbytes == 4) { int v; memcpy(&v, b, 4); return v; }
  if (nbytes == 1) return b[0];
  if (nbytes == 2) { short v; memcpy(&v, b, 2); return v; }
  long long v; memcpy(&v, b, 8); return v;
}
static float sys_read_f32(int fd) { float v = 0; readn(fd, &v, 4); return v; }
static string sys_read_str(int fd, unsigned n) { string s(n, '\0'); if (n) readn(fd, &s[0], n); return s; }
static CScriptArray* sys_read_bytes(int fd, unsigned n) {
  CScriptArray* arr = CScriptArray::Create(engine->GetTypeInfoByDecl("array<uint8>"), n);
  if (n) readn(fd, arr->GetBuffer(), n);
  return arr;
}
static bool writen(int fd, const void* buf, size_t n) {
  size_t sent = 0; while (sent < n) { ssize_t r = send(fd, (const char*)buf + sent, n - sent, MSG_NOSIGNAL); if (r <= 0) return false; sent += r; } return true;
}
static bool sys_write_int(int fd, long long v, int nbytes) { return writen(fd, &v, nbytes); }  // little endian host
static bool sys_write_f32(int fd, float v) { return writen(fd, &v, 4); }
static bool sys_write_f64(int fd, double v) { return writen(fd, &v, 8); }
static bool sys_write_str(int fd, const string& s) { return writen(fd, s.data(), s.size()); }
static bool sys_write_bytes(int fd, CScriptArray* a) { return writen(fd, a->GetBuffer(), a->GetSize()); }
static double sys_var(const string& name) { auto it = g_vars.find(name); return it == g_vars.end() ? 0.0 : it->second; }
static double sys_sqrt(double v) { return std::sqrt(v); }
static double sys_cos(double v) { return std::cos(v); }
static double sys_sin(double v) { return std::sin(v); }
static void sys_sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
static void StrLength(asIScriptGeneric* g) { auto* s = (string*)g->GetObject(); g->SetReturnDWord((asDWORD)s->size()); }

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: host stubs.as plugin.as port run_seconds [var=value ...]\n"); return 2; }
  engine = asCreateScriptEngine();
  engine->SetMessageCallback(asFUNCTION(MessageCallback), 0, asCALL_CDECL);
  engine->SetEngineProperty(asEP_PROPERTY_ACCESSOR_MODE, 3);
  RegisterStdString(engine);
  RegisterScriptArray(engine, true);
  engine->RegisterObjectMethod("string", "uint get_Length() const property", asFUNCTION(StrLength), asCALL_GENERIC);
  engine->RegisterObjectMethod("array<T>", "uint get_Length() const property", asMETHOD(CScriptArray, GetSize), asCALL_THISCALL);
  engine->RegisterGlobalFunction("uint64 sys_now_ms()", asFUNCTION(sys_now_ms), asCALL_CDECL);
  engine->RegisterGlobalFunction("void sys_log(const string&in)", asFUNCTION(sys_log), asCALL_CDECL);
  engine->RegisterGlobalFunction("int sys_listen(const string&in, int)", asFUNCTION(sys_listen), asCALL_CDECL);
  engine->RegisterGlobalFunction("int sys_accept(int)", asFUNCTION(sys_accept), asCALL_CDECL);
  engine->RegisterGlobalFunction("uint sys_avail(int)", asFUNCTION(sys_avail), asCALL_CDECL);
  engine->RegisterGlobalFunction("void sys_nodelay(int)", asFUNCTION(sys_nodelay), asCALL_CDECL);
  engine->RegisterGlobalFunction("void sys_close(int)", asFUNCTION(sys_close), asCALL_CDECL);
  engine->RegisterGlobalFunction("int64 sys_read_int(int, int)", asFUNCTION(sys_read_int), asCALL_CDECL);
  engine->RegisterGlobalFunction("float sys_read_f32(int)", asFUNCTION(sys_read_f32), asCALL_CDECL);
  engine->RegisterGlobalFunction("string sys_read_str(int, uint)", asFUNCTION(sys_read_str), asCALL_CDECL);
  engine->RegisterGlobalFunction("array<uint8>@ sys_read_bytes(int, uint)", asFUNCTION(sys_read_bytes), asCALL_CDECL);
  engine->RegisterGlobalFunction("bool sys_write_int(int, int64, int)", asFUNCTION(sys_write_int), asCALL_CDECL);
  engine->RegisterGlobalFunction("bool sys_write_f32(int, float)", asFUNCTION(sys_write_f32), asCALL_CDECL);
  engine->RegisterGlobalFunction("bool sys_write_f64(int, double)", asFUNCTION(sys_write_f64), asCALL_CDECL);
  engine->RegisterGlobalFunction("bool sys_write_str(int, const string&in)", asFUNCTION(sys_write_str), asCALL_CDECL);
  engine->RegisterGlobalFunction("bool sys_write_bytes(int, const array<uint8>&in)", asFUNCTION(sys_write_bytes), asCALL_CDECL);
  engine->RegisterGlobalFunction("double sys_var(const string&in)", asFUNCTION(sys_var), asCALL_CDECL);
  engine->RegisterGlobalFunction("double sys_sqrt(double)", asFUNCTION(sys_sqrt), asCALL_CDECL);
  engine->RegisterGlobalFunction("double sys_cos(double)", asFUNCTION(sys_cos), asCALL_CDECL);
  engine->RegisterGlobalFunction("double sys_sin(double)", asFUNCTION(sys_sin), asCALL_CDECL);
  engine->RegisterGlobalFunction("void sys_sleep_ms(int)", asFUNCTION(sys_sleep_ms), asCALL_CDECL);

  int port = atoi(argv[3]); double run_s = atof(argv[4]);
  g_vars["tmagent_port"] = port;
  for (int i = 5; i < argc; i++) { string a = argv[i]; auto p = a.find('='); if (p != string::npos) g_vars[a.substr(0, p)] = atof(a.substr(p + 1).c_str()); }

  auto slurp = [](const char* path) { std::ifstream f(path); std::stringstream ss; ss << f.rdbuf(); return ss.str(); };
  asIScriptModule* mod = engine->GetModule("m", asGM_ALWAYS_CREATE);
  string stubs = slurp(argv[1]), plugin = slurp(argv[2]);
  mod->AddScriptSection("stubs", stubs.c_str(), stubs.size());
  mod->AddScriptSection("plugin", plugin.c_str(), plugin.size());
  if (mod->Build() < 0) { printf("BUILD FAILED\n"); fflush(stdout); return 1; }
  printf("BUILD OK\n"); fflush(stdout);

  asIScriptContext* ctx = engine->CreateContext();
  auto call = [&](const char* decl) {
    asIScriptFunction* f = mod->GetFunctionByDecl(decl);
    if (!f) { printf("missing function %s\n", decl); exit(3); }
    ctx->Prepare(f);
    int r = ctx->Execute();
    if (r != asEXECUTION_FINISHED) {
      if (r == asEXECUTION_EXCEPTION) printf("EXCEPTION in %s: %s (line %d in %s)\n", decl, ctx->GetExceptionString(), ctx->GetExceptionLineNumber(), ctx->GetExceptionFunction()->GetName());
      else printf("execution of %s ended with %d\n", decl, r);
      fflush(stdout); exit(4);
    }
  };
  call("void HostInit()");
  call("void Main()");
  auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds((long long)(run_s * 1000));
  while (std::chrono::steady_clock::now() < end) call("void HostFrame()");
  printf("HOST DONE\n"); fflush(stdout);
  ctx->Release(); engine->ShutDownAndRelease();
  return 0;
}
