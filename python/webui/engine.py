# Persistent versor engine driver for the WebUI backend.
#
# Spawns one `versor-cli serve` subprocess and exchanges line-delimited
# JSON with it under a lock, so the engine keeps its transposition
# table warm across requests. If the engine dies, the next request
# transparently respawns it and raises EngineError so the API can
# return a 500 rather than hang.

import json
import os
import subprocess
import tempfile
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, binary=None, weights=None, mpc=None, mpc_t=2.0, tt_mb=128):
        self.binary = binary or os.environ.get(
            "VERSOR_BIN", os.path.join(REPO_ROOT, "target", "release", "versor-cli")
        )
        # models/pattern_best.bin is a symlink to the current best weights,
        # so a new best reaches the WebUI by repointing the link instead
        # of editing every place that names a weight file.
        self.weights = weights or os.environ.get(
            "VERSOR_WEIGHTS", os.path.join(REPO_ROOT, "models", "pattern_best.bin")
        )
        default_mpc = os.path.join(REPO_ROOT, "models", "mpc_params.txt")
        self.mpc = (
            mpc
            if mpc is not None
            else os.environ.get("VERSOR_MPC")
            or (default_mpc if os.path.exists(default_mpc) else None)
        )
        self.mpc_t = mpc_t
        self.tt_mb = tt_mb
        self._proc = None
        self._lock = threading.Lock()

    def _spawn(self):
        args = [self.binary, "serve", "--weights", self.weights, "--tt", str(self.tt_mb)]
        if self.mpc:
            args += ["--mpc", self.mpc, "--mpc-t", str(self.mpc_t)]
        # The engine writes to stderr only when it cannot start (for example
        # a missing or unreadable weight file); keep that line for the error.
        err = tempfile.TemporaryFile(mode="w+")
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                bufsize=1,
            )
        except OSError:
            err.close()
            raise
        # Wait for the readiness line.
        ready = proc.stdout.readline()
        if not ready or '"ready"' not in ready:
            proc.kill()
            proc.wait()
            err.seek(0)
            why = err.read().strip().splitlines()
            err.close()
            raise EngineError("engine did not report ready" + (": " + why[-1] if why else ""))
        err.close()
        self._proc = proc

    def _ensure(self):
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()

    def request(self, obj):
        with self._lock:
            try:
                self._ensure()
                line = json.dumps(obj, separators=(",", ":"))
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
                resp = self._proc.stdout.readline()
                if not resp:
                    raise EngineError("engine closed the connection")
                return json.loads(resp)
            except (BrokenPipeError, OSError, ValueError) as e:
                # Drop the dead engine; next request respawns it.
                if self._proc is not None:
                    try:
                        self._proc.kill()
                    except OSError:
                        pass
                self._proc = None
                raise EngineError(str(e))

    def close(self):
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.write('{"cmd":"quit"}\n')
                    self._proc.stdin.flush()
                except OSError:
                    pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    self._proc.kill()
                self._proc = None
