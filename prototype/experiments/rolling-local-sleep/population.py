"""
A Block is a group of neurons with its own lazy clock. Awake, it lives in VRAM and runs the GPU
kernels. Asleep, its words sit in (pinned) RAM and the CPU kernel sleeps them in place.

KernelPopulation wraps one Block with the same interface as PackedPopulation, and KernelNeuron
wraps a one-neuron KernelPopulation with the interface of PackedNeuron (for demo_single_neuron.run).
"""

import cupy as cp
import cupyx
import numpy as np

import common
from common import ELIG_STEPS, MAX_PENDING, USED_RATE, SeedStream, flow3, make_words
from cpu_sleep import cpu_sleep, sleep_neuron_state
import gpu_kernels as K

f32, i32 = np.float32, np.int32
FIELDS = ("words", "threshold", "plast", "post_hist", "age", "gate_bits")


class Block:
    def __init__(self, t, n, n_inputs, first_id, seed, xp=np, arrays=None, pinned=False):
        self.t, self.n, self.n_inputs, self.first_id = t, n, n_inputs, int(first_id)
        self.syn = t.n_branches * t.slots
        self.gstride = (t.n_branches + 7) // 8
        self.xp = xp
        self.seeds = SeedStream(seed)
        self.pending, self.fast_scale = 0, 1.0
        self.ew = tuple(f32(t.elig_decay ** k) for k in range(ELIG_STEPS))
        if arrays is None:
            arrays = self.empty_arrays(xp, pinned)
        for k, v in arrays.items():
            setattr(self, k, v)

    def empty_arrays(self, xp, pinned=False):
        n, t = self.n, self.t
        spec = {"words": ((n, self.syn), np.uint32), "threshold": ((n,), np.float32),
                "plast": ((n,), np.float16), "post_hist": ((n, ELIG_STEPS), np.float16),
                "age": ((n,), np.uint16), "gate_bits": ((n, self.gstride), np.uint8)}
        if xp is cp:
            return {k: cp.empty(s, d) for k, (s, d) in spec.items()}
        if pinned:
            return {k: cupyx.empty_pinned(s, d) for k, (s, d) in spec.items()}
        return {k: np.empty(s, d) for k, (s, d) in spec.items()}

    @classmethod
    def fresh(cls, t, n, n_inputs, rng, first_id=0, xp=np, pinned=False):
        b = cls(t, n, n_inputs, first_id, int(rng.integers(1 << 62)), xp=np, pinned=pinned and xp is np)
        b.words[:] = make_words(t, n, rng)
        b.threshold[:] = t.threshold_init
        b.plast[:] = t.plasticity_start
        b.post_hist[:] = 0
        b.age[:] = 0
        b.gate_bits[:] = np.packbits(np.ones((n, t.n_branches), bool), axis=1)
        return b.to(cp) if xp is cp else b

    def nbytes(self):
        return sum(getattr(self, k).nbytes for k in FIELDS)

    # ---------- moving between tiers ----------

    def to(self, xp, stream=None):
        """A copy of this block in the other tier (synchronous unless a stream is given)."""
        other = Block(self.t, self.n, self.n_inputs, self.first_id, 0, xp=xp)
        other.seeds, other.pending, other.fast_scale = self.seeds, self.pending, self.fast_scale
        for k in FIELDS:
            a = getattr(self, k)
            setattr(other, k, cp.asarray(a) if xp is cp else cp.asnumpy(a, stream=stream))
        return other

    def copy_into(self, dst, stream):
        """Asynchronous copy of all arrays into an existing block in the other tier."""
        for k in FIELDS:
            s, d = getattr(self, k), getattr(dst, k)
            if isinstance(d, cp.ndarray):
                d.set(s, stream=stream)
            else:
                s.get(stream=stream, out=d, blocking=False)  # async: d is pinned
        dst.pending, dst.fast_scale, dst.seeds = self.pending, self.fast_scale, self.seeds
        dst.first_id = self.first_id

    # ---------- awake (GPU) ----------

    def forward(self, x_dev, rate_out, context=0.0, learn_trace=True, stream=None):
        t = self.t
        m = flow3(t, self.pending)[0]
        args = (self.words, x_dev, i32(self.n_inputs), np.uint64(self.first_id), self.gate_bits,
                i32(self.gstride), self.threshold, self.post_hist, rate_out,
                f32(m[0]), f32(m[1]), f32(m[2]), f32(self.fast_scale), f32(t.w_max / 31),
                f32(t.branch_threshold), f32(t.spike_bonus), f32(1.0 + t.context_gain * context),
                f32(t.soma_slope), f32(t.threshold_lr if learn_trace else 0.0), f32(t.target_rate),
                f32(t.sign), i32(learn_trace), f32(USED_RATE))
        K.forward_k((self.n,), (self.syn,), args, stream=stream)

    def modulate(self, pre0, pstride, dopamine=0.0, ach=0.0, arousal=0.0, stream=None):
        if dopamine == 0.0:
            return
        t = self.t
        oneshot = abs(dopamine) * ach > t.oneshot_level
        P = flow3(t, self.pending)
        args = (self.words, i32(self.n_inputs), np.uint64(self.first_id), self.gate_bits,
                i32(self.gstride), self.plast, self.post_hist, pre0, i32(pstride), *self.ew,
                f32(t.lr), f32(ach), f32(dopamine), f32(arousal * abs(dopamine)),
                f32(self.fast_scale), f32(t.w_max / 31), f32(t.w_max), f32(31 / t.w_max),
                i32(oneshot), *K.mat9(P), np.uint32(self.seeds.next() & 0xFFFFFFFF))
        K.modulate_k((self.n,), (self.syn,), args, stream=stream)
        self.fast_scale = 1.0
        if oneshot:
            self.pending = 0

    def catch_up(self, stream=None):
        if not self.pending:
            return
        t, nw = self.t, self.n * self.syn
        args = (self.words, np.int64(nw), f32(t.w_max / 31), f32(t.w_max), f32(31 / t.w_max),
                *K.mat9(flow3(t, self.pending)), np.uint32(self.seeds.next() & 0xFFFFFFFF))
        K.catchup_k(((nw + 255) // 256,), (256,), args, stream=stream)
        self.pending = 0

    def tick(self, stream=None):
        self.pending += 1
        self.fast_scale *= self.t.fast_decay
        if self.pending >= MAX_PENDING:
            self.catch_up(stream)

    def rest(self, steps, stream=None):
        self.pending += steps
        self.fast_scale *= self.t.fast_decay ** steps
        self.catch_up(stream)

    # ---------- asleep ----------

    def sleep_gpu(self, counter=None, stream=None):
        """GPU sleep of the whole block, in place (for the synchronous / micro-sleep comparisons)."""
        t = self.t
        nw = self.n * self.syn
        counter = cp.zeros(1, cp.uint32) if counter is None else counter
        args = (self.words, np.int64(nw), i32(self.pending > 0), *K.mat9(flow3(t, self.pending)),
                *K.mat9(flow3(t, t.sleep_flow_steps)), f32(self.fast_scale), f32(t.w_max / 31),
                f32(t.w_max), f32(31 / t.w_max), f32(1 - t.downscale), f32(t.prune_eps),
                f32(t.tag_decay), np.uint32(self.seeds.next() & 0xFFFFFFFF), counter)
        K.sleep_k(((nw + 255) // 256,), (256,), args, stream=stream)
        with (stream or cp.cuda.Stream.null):
            self.post_hist[:] = 0
            self.age += 1
            self.plast[:] = (t.plasticity_floor + (self.plast - t.plasticity_floor) * t.maturation
                             ).astype(cp.float16)
        self.pending, self.fast_scale = 0, 1.0
        return counter

    def sleep_cpu(self, threads=None, impl="c"):
        """CPU sleep, in place on host arrays (this block must live in RAM)."""
        kw = {} if threads is None else {"threads": threads}
        n = cpu_sleep(self.words, self.t, self.pending, self.fast_scale, self.seeds.next(), impl=impl, **kw)
        sleep_neuron_state(self.t, self.plast, self.age, self.post_hist)
        self.pending, self.fast_scale = 0, 1.0
        return n


class KernelPopulation:
    """PackedPopulation interface on top of one GPU Block. Sleep runs on the CPU (default) or GPU."""

    def __init__(self, ntype, n_neurons, n_inputs, rng, first_id=0, sleep_on="cpu", block=None):
        self.t, self.n, self.n_inputs = ntype, n_neurons, n_inputs
        self.sleep_on = sleep_on
        self.b = block if block is not None else Block.fresh(ntype, n_neurons, n_inputs, rng, first_id, xp=cp)
        self.pre_hist = np.zeros((ELIG_STEPS, n_inputs), np.float16)
        self.rate = cp.zeros(n_neurons, cp.float32)

    @classmethod
    def from_packed(cls, pp, seed, sleep_on="cpu"):
        """Same initial state as a prototype PackedPopulation (for exact-start comparisons)."""
        b = Block(pp.t, pp.n, pp.n_inputs, pp.first_id, seed, xp=np)
        b.words = pp.words.reshape(pp.n, -1).copy()
        b.threshold = pp.threshold.copy()
        b.plast = pp.plasticity.copy()
        b.post_hist = pp.post_hist.copy()
        b.age = pp.age.copy()
        b.gate_bits = pp.gate_bits.copy()
        b.pending, b.fast_scale = pp.pending, pp.fast_scale
        return cls(pp.t, pp.n, pp.n_inputs, None, sleep_on=sleep_on, block=b.to(cp))

    def forward(self, x, context=0.0, learn_trace=True):
        self.b.forward(cp.asarray(x, cp.float32), self.rate, context, learn_trace)
        if learn_trace:
            self.pre_hist = np.roll(self.pre_hist, 1, axis=0)
            self.pre_hist[0] = x
        return self.rate.get() * 1.0

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        pre = cp.asarray(self.pre_hist)
        self.b.modulate(pre, self.n_inputs, dopamine, ach, arousal)

    def tick(self):
        self.b.tick()

    def rest(self, steps):
        self.b.rest(steps)

    def sleep(self):
        if self.sleep_on == "gpu":
            n = int(self.b.sleep_gpu().get()[0])
        else:
            h = self.b.to(np)
            n = h.sleep_cpu()
            self.b = h.to(cp)
        self.pre_hist[:] = 0
        return n

    def words(self):
        return self.b.words.get()


class KernelNeuron:
    """PackedNeuron interface, so demo_single_neuron.run(..., make=KernelNeuron) works."""

    sleep_on = "cpu"

    def __init__(self, ntype, n_inputs, rng):
        # consume the shared rng exactly like PackedNeuron + PackedPopulation.__init__ do
        from packed_neuron import PackedPopulation
        pp = PackedPopulation(ntype, 1, n_inputs, rng, first_id=int(rng.integers(1 << 31)))
        self.pop = KernelPopulation.from_packed(pp, seed=pp.first_id * 7919 + 1, sleep_on=self.sleep_on)

    def forward(self, x, context=0.0, learn_trace=True):
        return float(self.pop.forward(x, context, learn_trace)[0])

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        self.pop.modulate(dopamine, ach, arousal)

    def tick(self):
        self.pop.tick()

    def rest(self, steps):
        self.pop.rest(steps)

    def sleep(self):
        return self.pop.sleep()

    def memory_bytes(self):
        return self.pop.b.nbytes()


class KernelNeuronGpuSleep(KernelNeuron):
    sleep_on = "gpu"
