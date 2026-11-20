"""Standard library client. Supply a command that starts the bridge."""
import json
import subprocess


class GameClient:
    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8")
        try:
            self.ready = self._read()
            if self.ready.get("type") != "ready" or self.ready.get("version") != 1:
                raise RuntimeError("unexpected bridge greeting")
        except BaseException:
            self.close()
            raise
        self.turn = None

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise EOFError("bridge closed its output")
        reply = json.loads(line)
        if reply.get("type") == "error":
            raise ValueError(reply["message"])
        return reply

    def request(self, **message):
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self._read()

    def observe(self, image, **fields):
        reply = self.request(type="observe", image=image, **fields)
        self.turn = reply["turn"]
        return reply["action"]

    def outcome(self, reward, **fields):
        self.request(type="outcome", turn=self.turn, reward=reward, **fields)
        self.turn = None

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.process.stdout.close()
        return code

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        if self.close() and kind is None:
            raise RuntimeError("bridge exited with an incomplete session")
