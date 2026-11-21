"""
Design stamps for saved brains (note 79, plan phase 0).

Every brain made by `prepare()` carries `brain.design`:
  kind       which brain description built it ("full", "vision", "human100k", ...)
  spec_hash  a fingerprint of its whole design: every group, cell-type parameter and connection
  runtime    RUNTIME_VERSION: bump it when the simulation's rules change meaning (how a step, learning or sleep
             works), so learned state saved under the old rules is not silently misread
  code       the code version (git) that made it
  created    when it was made (UTC, ISO 8601)

`check(brain, kind)` decides whether a loaded brain may be used:
  refused  a different runtime version; the wrong kind (when a kind is expected); a design that does not match its own
           fingerprint (a damaged or edited file)
  warning  a save from before stamps existed; a design that differs from what the current code builds for that kind
           (the brain still carries and runs its own design)
"""

import dataclasses
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

RUNTIME_VERSION = 1   # bump when step / learning / sleep semantics change incompatibly


class DesignError(ValueError):
    """A saved brain that must not be loaded here."""


def _plain(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, float):
        return round(x, 9)
    if hasattr(x, "item"):          # numpy scalars
        return _plain(x.item())
    return x


def spec_hash(spec):
    """A fingerprint of a brain description: groups (with cell-type parameters), connections and inputs."""
    record = {"groups": _plain(spec.groups), "synaptic": _plain(spec.synaptic), "pooled": _plain(spec.pooled),
              "inputs": _plain(spec.inputs), "extra": _plain({k: v for k, v in vars(spec).items()
                                                               if k not in ("groups", "synaptic", "pooled", "inputs")})}
    text = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def code_version():
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def stamp(brain, kind):
    brain.design = {"kind": str(kind), "spec_hash": spec_hash(brain.spec), "runtime": RUNTIME_VERSION,
                    "code": code_version(),
                    "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    return brain


def current_hash(kind):
    """The fingerprint the current code would build for `kind`, or None if unknown."""
    try:
        import brainspec as S
        from run_signal_game import SPECS
        return spec_hash(getattr(S, SPECS[kind])())
    except Exception:
        return None


def check(brain, kind=None):
    """Returns a list of warnings; raises DesignError if the brain must not be used."""
    d = getattr(brain, "design", None)
    if not d:
        return ["this brain was saved before design stamps existed (note 79); it can't be checked"]
    if d.get("runtime") != RUNTIME_VERSION:
        raise DesignError(f"this brain was made with runtime version {d.get('runtime')}, but this program runs "
                          f"version {RUNTIME_VERSION}: its learned state would be misread. Grow a new brain or use a "
                          f"matching program version (made by code {d.get('code')}).")
    if spec_hash(brain.spec) != d.get("spec_hash"):
        raise DesignError("this brain's design does not match its own fingerprint: the file is damaged or was edited")
    if kind and d.get("kind") != kind:
        raise DesignError(f"this is a '{d.get('kind')}' brain, but a '{kind}' brain was expected")
    warnings = []
    now = current_hash(d.get("kind"))
    if now is not None and now != d.get("spec_hash"):
        warnings.append(f"the '{d.get('kind')}' design has changed since this brain was made ({d.get('created')}, "
                        f"code {d.get('code')}); it still runs with its own design")
    return warnings


def describe(brain):
    d = getattr(brain, "design", None)
    if not d:
        return "unversioned brain"
    return f"{d['kind']} brain, design {d['spec_hash']}, runtime v{d['runtime']}, made {d['created']} (code {d['code']})"
