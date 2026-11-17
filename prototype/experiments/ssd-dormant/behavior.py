"""
Behavior check: does the neuron still behave like PackedNeuron when
  (a) it is computed by our Numba kernels on raw packed words ("eager"), and
  (b) it goes DORMANT: nights with no activity are skipped, composed into one 3x3 map, its words
      are zstd-compressed onto the SSD and dropped from RAM, and the next time it is needed it is
      RECALLED (read back, decompressed, all skipped nights applied at once) ("lazy+dormant")?

Same story as demo_single_neuron.py: childhood -> hard times -> 6 quiet days -> relearn.
During the 6 quiet days the dormant neuron lives on the SSD and is only recalled for the probes.
"""

import os
from dataclasses import replace

from common import DATA, LEDGER, drop_cache, save

import numpy as np
import zstandard as zstd

from demo_single_neuron import run
from kernels import (deferred_night, flow_matrix_3, k_checksum, k_forward, k_init, k_modulate,
                     k_sleep, night_matrices)
from neuron import WORKHORSE
from packed_neuron import ELIG_STEPS, MAX_PENDING, PackedNeuron

ROW0 = np.zeros(1, np.int64)
F32 = np.float32


class SSDNeuron:
    """Same interface as PackedNeuron (forward / modulate / tick / rest / sleep / memory_bytes)."""

    def __init__(self, ntype, n_inputs, rng, lazy=False, dormant_after=None):
        t = self.t = ntype
        self.n_in, self.rng, self.lazy, self.dormant_after = n_inputs, rng, lazy, dormant_after
        self.id = int(rng.integers(1 << 31))
        self.ids = np.array([self.id], np.int64)
        self.words = np.zeros((1, 256), np.uint32)
        k_init(self.words, self.id, self._seed())     # unused cascade levels start (and stay) at 0
        self.gate = np.full(1, 255, np.uint8)
        self.thresh = np.full(1, t.threshold_init, F32)
        self.plast = np.full(1, t.plasticity_start, F32)
        self.post_hist = np.zeros((1, ELIG_STEPS), F32)
        self.pre_hist = np.zeros((ELIG_STEPS, n_inputs), F32)
        self.elig_w = (t.elig_decay ** np.arange(ELIG_STEPS)).astype(F32)
        self.pending, self.fast_scale = 0, 1.0
        self.touched = False             # any learning since the last real night?
        self.M = np.eye(3)               # skipped nights, composed
        self.skipped = 0
        self.file = None                 # set while the neuron is dormant on the SSD
        self.recalls = 0

    def _seed(self):
        return np.uint64(self.rng.integers(1 << 63))

    # ---- dormancy

    def _evict(self):
        before = int(k_checksum(self.words))
        blob = zstd.ZstdCompressor(level=3).compress(self.words.tobytes())
        self.file = DATA / f"dormant_{os.getpid()}_{self.id}.zst"
        with open(self.file, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        LEDGER.add(len(blob))
        drop_cache(str(self.file))
        self.words, self._check = None, before

    def _recall(self):
        if self.file is not None:
            raw = zstd.ZstdDecompressor().decompress(self.file.read_bytes())
            self.words = np.frombuffer(raw, np.uint32).reshape(1, 256).copy()
            assert int(k_checksum(self.words)) == self._check, "dormant round trip is not bit-exact"
            self.file.unlink()
            self.file = None
            self.recalls += 1
        if self.skipped:
            t = self.t
            k_sleep(self.words, ROW0, self.ids, self.M.astype(F32), F32(0.0), np.eye(3, dtype=F32),
                    F32(1.0), F32(t.prune_eps), F32(t.tag_decay ** self.skipped), F32(t.w_max),
                    self._seed(), True, np.zeros(1, np.int64))
            self.M, self.skipped = np.eye(3), 0

    def _catch_up(self):
        if self.pending:
            M = flow_matrix_3(self.t, self.pending).astype(F32)
            k_sleep(self.words, ROW0, self.ids, M, F32(0), np.eye(3, dtype=F32), F32(1), F32(0), F32(1),
                    F32(self.t.w_max), self._seed(), False, np.zeros(1, np.int64))
            self.pending = 0

    # ---- interface

    def forward(self, x, context=0.0, learn_trace=True):
        self._recall()
        t = self.t
        rate = np.zeros(1, F32)
        F0 = flow_matrix_3(t, self.pending)[0].astype(F32)
        k_forward(self.words, ROW0, self.ids, self.id, self.gate, self.thresh, self.post_hist,
                  x.astype(F32), self.n_in, 0, F0, F32(self.fast_scale), F32(t.w_max),
                  F32(t.branch_threshold), F32(t.spike_bonus), F32(1.0 + t.context_gain * context),
                  F32(t.soma_slope), F32(t.threshold_lr), F32(t.target_rate), learn_trace, rate)
        if learn_trace:
            self.touched = True
            self.pre_hist = np.roll(self.pre_hist, 1, axis=0)
            self.pre_hist[0] = x
        return float(rate[0]) * t.sign

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        if dopamine == 0.0:
            return
        self._recall()
        t = self.t
        self.touched = True
        oneshot = abs(dopamine) * ach > t.oneshot_level
        if oneshot:
            self._catch_up()
        k_modulate(self.words, ROW0, self.ids, self.id, self.gate, self.plast, self.post_hist,
                   self.pre_hist, self.elig_w, self.n_in, 0, F32(self.fast_scale), F32(t.w_max),
                   F32(t.lr), F32(dopamine), F32(ach), F32(arousal), oneshot, self._seed())
        self.fast_scale = 1.0

    def tick(self):
        self.pending += 1
        self.fast_scale *= self.t.fast_decay
        if self.pending >= MAX_PENDING and self.words is not None:
            self._catch_up()

    def rest(self, steps):
        self.pending += steps
        self.fast_scale *= self.t.fast_decay ** steps
        if not self.skipped:          # a dormant neuron folds rests into its skipped-night map
            self._catch_up()

    def sleep(self):
        t = self.t
        n = 0
        if self.lazy and not self.touched:
            # Nothing happened today: skip the night, remember it as one more 3x3 map.
            self.M = deferred_night(t, self.M, self.pending)
            self.pending, self.skipped = 0, self.skipped + 1
            if self.dormant_after and self.skipped >= self.dormant_after and self.file is None:
                self._evict()
        else:
            self._recall()
            Mpre, Ms, keep0 = night_matrices(t, self.pending)
            pr = np.zeros(1, np.int64)
            k_sleep(self.words, ROW0, self.ids, Mpre, F32(self.fast_scale), Ms, keep0, F32(t.prune_eps),
                    F32(t.tag_decay), F32(t.w_max), self._seed(), True, pr)
            n = int(pr[0])
            self.pending = 0
        self.touched = False
        self.fast_scale = 1.0
        self.pre_hist[:] = 0
        self.post_hist[:] = 0
        self.plast[:] = t.plasticity_floor + (self.plast - t.plasticity_floor) * t.maturation
        return n

    def memory_bytes(self):
        soma = self.gate.nbytes + self.thresh.nbytes + self.plast.nbytes + self.post_hist.nbytes
        return soma + (0 if self.words is None else self.words.nbytes)


def summarize(rows):
    cols = ["after hard times (A bad)", "time passes: day 3", "time passes: day 6"]
    out = {}
    for c in cols:
        v = np.array([dict(log)[c] for log, _ in rows])
        out[c] = (float(v.mean()), float(v.min()), float(v.max()))
    r = np.array([x for _, x in rows])
    out["relearn"] = (float(np.median(r)), int(r.min()), int(r.max()))
    return out


def main(seeds=range(1, 9)):
    makers = {
        "packed (baseline)": PackedNeuron,
        "ssd kernels, eager nights": lambda t, n, r: SSDNeuron(t, n, r),
        "ssd kernels, lazy nights + dormant on SSD": lambda t, n, r: SSDNeuron(t, n, r, lazy=True,
                                                                              dormant_after=1),
    }
    res = {}
    print("== Behavior (seeds 1-8): mean [min-max]; relearn trials median [min-max] ==")
    print(f"  {'':<52}{'hard times':>18}{'day 3':>18}{'day 6':>18}{'relearn':>16}")
    for label, ntype in [("workhorse", WORKHORSE), ("plain", replace(WORKHORSE, cascade_levels=1))]:
        for impl, make in makers.items():
            rows = [run("", ntype, seed=s, make=make, verbose=False) for s in seeds]
            sm = summarize(rows)
            res[f"{label} / {impl}"] = sm
            cells = [f"{m:.2f} [{a:.2f}-{b:.2f}]" for m, a, b in
                     (sm[c] for c in ["after hard times (A bad)", "time passes: day 3", "time passes: day 6"])]
            print(f"  {label + ' / ' + impl:<52}" + "".join(f"{c:>18}" for c in cells)
                  + f"{sm['relearn'][0]:>7.0f} [{sm['relearn'][1]}-{sm['relearn'][2]}]")
    save("behavior", res)
    return res


if __name__ == "__main__":
    main()
