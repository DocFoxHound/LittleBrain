"""
Cortex reliability and honest memory, one configuration (notes/28).

  .venv/bin/python prototype/brain/run_cortex_fix.py NAME L4_TARGET FF_GAIN REC_GAIN
    L4_TARGET  L4 target firing rate (0 = cell type default)
    FF_GAIN    L4 -> L2/3 synapse strength (1 = as before)
    REC_GAIN   L2/3 -> L2/3 basal synapse strength
  env: SEED, HOMEO_RULE=quantile|log, LRF (learning-rate factor for excitatory cells), FR (fast-weight cap)

Prints reliability (rate/retest per group) after calibration and after 4 development days, then the scrambled
memory test against an untrained control.
"""
import sys, os, copy, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import brainspec as S
import diagnostics as D
from dataclasses import replace
from federation import Brain
from run_wiring_test import blobs, N_PATTERNS
G = ["TH.TC", "A.L4", "A.L23", "A.L5IT", "A.L6", "B.L4", "B.L23", "B.L5IT"]


def rel(b):
    rng = np.random.default_rng(9)
    pats = np.array([blobs(b.pos["retina"], rng) for _ in range(N_PATTERNS)])
    return D.reliability(copy.deepcopy(b), pats, G)


def fmt(r):
    return " ".join(f"{k.replace('TH.', '')}:{v['rate']:.2f}/{v['retest']:+.2f}" for k, v in r.items())


v=sys.argv[1]; l4t=float(sys.argv[2]); ffg=float(sys.argv[3]); recg=float(sys.argv[4]); seed=int(os.environ.get("SEED","0"))
spec=S.stage1()
if l4t: spec.groups=[replace(gr, ctype=replace(gr.ctype, target_rate=l4t)) if gr.key.endswith(".L4") else gr for gr in spec.groups]
syn=[]
for p in spec.synaptic:
    if p.kind=="local" and p.src.endswith(".L4") and p.dst.endswith(".L23"): p=replace(p,gain=ffg)
    if p.kind=="local" and p.src.endswith(".L23") and p.dst.endswith(".L23") and p.comp=="basal": p=replace(p,gain=recg)
    syn.append(p)
spec.synaptic=syn
lrf=float(os.environ.get("LRF","1")); fr=float(os.environ.get("FR","0"))
if lrf!=1 or fr:
    from cells import ExcitatoryGroup
    def tweak(ct):
        if ct is None or ct.family in ("interneuron",) or ct.sign<0: return ct
        r=ct.rules; r=replace(r, lr=r.lr*lrf, fast_range=fr or r.fast_range); return replace(ct, rules=r)
    spec.groups=[replace(gr, ctype=tweak(gr.ctype)) if gr.ctype is not None and hasattr(gr.ctype,"rules") else gr for gr in spec.groups]
t0=time.time()
b=Brain(spec,seed=seed); b.homeostasis=0.02
rng=np.random.default_rng(1000+seed)
b.calibrate(lambda: {"retina": (rng.random(len(b.pos["retina"])) < 0.2).astype(np.float32)})
print(f"{v:<12} cal  {fmt(rel(b))}",flush=True)
b.develop(lambda: {"retina": blobs(b.pos["retina"], rng)})
print(f"{v:<12} dev4 {fmt(rel(b))}",flush=True)
m=D.memory_vs_control(b,seed); L,U=m["learned"],m["untrained"]
print(f"{v:<12} seed{seed} memory match {L['match']:+.3f} vs ctl {U['match']:+.3f} diff {L['match']-U['match']:+.3f} | id {L['accuracy']:.2f} vs ctl {U['accuracy']:.2f} ({time.time()-t0:.0f}s)")
