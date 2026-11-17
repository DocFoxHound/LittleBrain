"""
The packed workhorse population (prototype/packed_neuron.py) on the GPU, with the synapse words
stored wherever we like:

    resident      all words in VRAM (the "it fits" case)
    paged         words in pinned host RAM; a VRAM cache of K blocks, LRU, async copies on
                  CUDA streams overlapping compute (virtual memory for a brain)
    zerocopy      words in pinned, mapped host RAM; kernels read them straight over PCIe
    managed       CUDA managed memory; with cache_blocks=K we keep at most K blocks on the GPU
                  using prefetch hints (driver does the copying); with cache_blocks=0 the GPU
                  reads host-resident pages remotely (preferred location = host)
    managed_lazy  CUDA managed memory, no hints at all: pages migrate on GPU page faults

Small per-neuron state (threshold, plasticity, firing history, gate, id: ~25 bytes) always lives
in VRAM. Only the 1 KB of synapse words per neuron is paged.

Semantics are those of PackedPopulation, with one change that makes paging possible: the two
"shared clocks" (pending cascade flow, fast-weight decay) are kept *per block* instead of per
population. A block that is not touched is not touched at all, not even by modulate's decay
bake or tick's periodic catch-up. The math is identical; only when stochastic rounding happens
differs (less often, so slightly less rounding noise).
"""

import ctypes
import os
import sys
from functools import lru_cache

import numpy as np
import cupy as cp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import packed_neuron as pn  # noqa: E402
from neuron import NeuronType  # noqa: E402

KERNELS = ("prep_kernel", "forward_kernel", "quiet_kernel", "modulate_kernel", "sleep_kernel",
           "init_kernel", "input_kernel")
MODES = ("resident", "paged", "zerocopy", "managed", "managed_lazy")
H2D, D2H = 1, 2


def _lit(v):
    return f"{float(np.float32(v))!r}f"


@lru_cache(maxsize=8)
def kernels(t: NeuronType):
    assert t.slots == 32, "kernel maps one branch's 32 slots onto one warp"
    assert t.n_branches <= 8, "gate bits are one byte per neuron"
    assert t.cascade_levels <= 3
    d = {
        "NB": t.n_branches, "DEC": _lit(t.w_max / pn.Q), "ENCF": _lit(pn.Q / t.w_max), "WMAX": _lit(t.w_max),
        "BRANCH_THR": _lit(t.branch_threshold), "SPIKE_BONUS": _lit(t.spike_bonus),
        "SOMA_SLOPE": _lit(t.soma_slope), "THR_LR": _lit(t.threshold_lr), "TARGET_RATE": _lit(t.target_rate),
        "SIGN": _lit(t.sign), "USED_RATE": _lit(pn.USED_RATE), "LR_H": _lit(np.float16(t.lr)),
        "DOWNSCALE": _lit(t.downscale), "PRUNE_EPS": _lit(t.prune_eps), "TAG_DECAY": _lit(t.tag_decay),
    }
    for k in range(4):
        d[f"EW{k}"] = _lit(t.elig_decay ** k)
    src = "".join(f"#define {k} {v}\n" for k, v in d.items()) + open(os.path.join(HERE, "kernels.cu")).read()
    mod = cp.RawModule(code=src, options=("-std=c++17",))
    return {name: mod.get_function(name) for name in KERNELS}


def _step3(t: NeuronType) -> np.ndarray:
    """One awake step of cascade flow as a 3x3 float64 matrix (unused levels: identity on zero)."""
    L = t.cascade_levels
    cap = 2.0 ** np.arange(L)
    g = t.cascade_g * 2.0 ** -np.arange(max(L - 1, 0))
    step = np.zeros((L, L))
    for col in range(L):
        c = np.eye(L)[:, col].copy()
        for k in range(L - 1):
            f = g[k] * (c[k] - c[k + 1])
            c[k] -= f / cap[k]
            c[k + 1] += f / cap[k + 1]
        c[-1] *= 1.0 - t.deep_leak
        step[:, col] = c
    m = np.eye(3)
    m[:L, :L] = step
    return m


def _pow_table(t: NeuronType) -> np.ndarray:
    tab = np.zeros((31, 3, 3))
    tab[0] = _step3(t)
    for k in range(1, 31):
        tab[k] = tab[k - 1] @ tab[k - 1]
    return tab


# ---- CUDA driver API (managed-memory hints; the runtime wrappers reject host locations in CUDA 13)
class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


_cu = None


def _driver():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL("libcuda.so.1")
        _cu.cuMemAdvise_v2.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, _Loc]
        _cu.cuMemPrefetchAsync_v2.argtypes = [ctypes.c_uint64, ctypes.c_size_t, _Loc, ctypes.c_uint, ctypes.c_void_p]
    return _cu


LOC_DEV, LOC_HOST = _Loc(1, 0), _Loc(2, 0)
ADV_PREFERRED, ADV_ACCESSED_BY = 3, 5


def mem_advise(ptr, size, advice, loc):
    r = _driver().cuMemAdvise_v2(ptr, size, advice, loc)
    assert r == 0, f"cuMemAdvise failed: {r}"


def prefetch(ptr, size, loc, stream):
    r = _driver().cuMemPrefetchAsync_v2(ptr, size, loc, 0, ctypes.c_void_p(stream.ptr))
    assert r == 0, f"cuMemPrefetchAsync failed: {r}"


def pinned_array(nbytes, dtype=np.uint32):
    mem = cp.cuda.PinnedMemory(nbytes, 3)  # portable | mapped (zero-copy capable)
    ptr = cp.cuda.PinnedMemoryPointer(mem, 0)
    arr = np.frombuffer(ptr, dtype, nbytes // np.dtype(dtype).itemsize)
    return arr, mem


class GPUPopulation:
    def __init__(self, ntype: NeuronType, n_neurons: int, *, n_inputs=None, col_size=None, npc=None,
                 block_size=1024, mode="resident", cache_blocks=0, chunk_blocks=8, seed=0, ids=None,
                 init=True):
        assert mode in MODES
        t = self.t = ntype
        self.k = kernels(t)
        self.mode, self.N, self.bs = mode, int(n_neurons), int(block_size)
        self.nb = -(-self.N // self.bs)
        self.wpn = t.n_branches * 32
        self.block_words = self.bs * self.wpn
        self.block_bytes = self.block_words * 4
        if col_size is None:  # flat wiring: every neuron draws from the whole input space (the baseline)
            col_size, npc = n_inputs, 0xFFFFFFFF
        self.col_size, self.npc = int(col_size), int(npc)
        self.col_mask = self.col_size - 1 if self.col_size & (self.col_size - 1) == 0 else 0
        if n_inputs is None:
            n_inputs = -(-self.N // self.npc) * self.col_size
        self.n_inputs = int(n_inputs)
        self.seed = int(seed) * 1_000_003 + 17

        # blocking streams: they also order against work issued on the legacy default stream
        self.s_comp = cp.cuda.Stream(non_blocking=False)
        self.s_in = cp.cuda.Stream(non_blocking=False)
        self.s_out = cp.cuda.Stream(non_blocking=False)
        N = self.N
        with self.s_comp:
            # per neuron, always in VRAM (~25 bytes)
            self.ids = cp.arange(N, dtype=cp.uint32) if ids is None else cp.asarray(ids, cp.uint32)
            self.gate = cp.full(N, (0xFF << (8 - t.n_branches)) & 0xFF, cp.uint8)
            self.thr = cp.full(N, t.threshold_init, cp.float32)
            self.plast = cp.full(N, t.plasticity_start, cp.float16)
            self.post = cp.zeros((N, 4), cp.float16)  # ring buffer of recent firing
            self.age = cp.zeros(N, cp.uint16)
            self.rate = cp.zeros(N, cp.float32)
            # per block
            self.flow_clock = cp.zeros(self.nb, cp.int32)
            self.fast_clock = cp.zeros(self.nb, cp.int32)
            self.bcoef = cp.zeros(self.nb * 16, cp.float32)
            self.blk_active = cp.zeros(self.nb, cp.uint8)
            # shared
            self.x = cp.zeros(self.n_inputs, cp.float32)
            self.pre = cp.zeros((4, self.n_inputs), cp.float16)  # ring buffer of recent input
            self.mpow = cp.asarray(_pow_table(t).reshape(-1))
            self.counters = cp.zeros(3, cp.uint64)  # changed words, pruned synapses, 128 B lines written
            self.sleep_m = [float(v) for v in np.eye(3, dtype=np.float32).reshape(-1)]
            sm = np.eye(3, dtype=np.float32)
            L = t.cascade_levels
            sm[:L, :L] = pn._flow_matrix(t, t.sleep_flow_steps)
            self.sleep_m = [float(v) for v in sm.reshape(-1)]
        self.log_decay = float(np.log(t.fast_decay))
        self.T, self.ring, self.fwd_count, self.op = 0, 0, 0, 0
        self.last_active = np.full(self.nb, -(1 << 40), np.int64)
        self.x_cols, self.pre_cols = None, [None] * 4
        self.h2d = self.d2h = 0
        self._alloc_words(cache_blocks, chunk_blocks)
        if init:
            self._init_words()

    # ------------------------------------------------------------------ storage
    def _alloc_words(self, K, chunk):
        nbytes = self.nb * self.block_bytes
        self.K = int(K)
        self.host = None
        if self.mode == "resident":
            self.dwords = cp.empty(self.nb * self.block_words, cp.uint32)
            self.base = self.dwords.data.ptr
        elif self.mode in ("paged", "zerocopy"):
            self.host, self._pin = pinned_array(nbytes)
            self.base = self.host.ctypes.data  # with UVA the host pointer is valid on the device
        else:
            self._mm = cp.cuda.malloc_managed(nbytes)
            self.base = self._mm.ptr
            self.host = np.ctypeslib.as_array((ctypes.c_uint32 * (nbytes // 4)).from_address(self.base))
            self.host[:] = 0  # first touch on the CPU: pages start in RAM
            if self.mode == "managed":
                mem_advise(self.base, nbytes, ADV_PREFERRED, LOC_HOST)
                mem_advise(self.base, nbytes, ADV_ACCESSED_BY, LOC_DEV)
        if self.K > 0:
            assert self.K >= 2 and self.mode != "resident"
            self.chunk = max(1, min(chunk, self.K // 2))
            if self.mode in ("paged", "zerocopy"):
                self.cache = cp.empty(self.K * self.block_words, cp.uint32)
                self.slot_ptr = (self.cache.data.ptr + np.arange(self.K, dtype=np.uint64) *
                                 np.uint64(self.block_bytes)).astype(np.uint64)
                self.slot_dirty = cp.zeros(self.K, cp.uint32)
            self.slot_of = np.full(self.nb, -1, np.int64)
            self.block_in = np.full(self.K, -1, np.int64)
            self.slot_key = np.full(self.K, -1, np.int64)

    def switch_mode(self, mode):
        """Run the same (host-backed) population as resident / paged / zerocopy."""
        if mode == self.mode:
            return
        assert {mode, self.mode} <= {"resident", "paged", "zerocopy"} and self.host is not None
        self.flush()
        nbytes = self.nb * self.block_bytes
        if mode == "resident":
            if getattr(self, "dwords", None) is None:
                self.dwords = cp.empty(self.nb * self.block_words, cp.uint32)
            cp.cuda.runtime.memcpy(self.dwords.data.ptr, self.host.ctypes.data, nbytes, H2D)
            self.base = self.dwords.data.ptr
        else:
            if self.mode == "resident":
                cp.cuda.runtime.memcpy(self.host.ctypes.data, self.dwords.data.ptr, nbytes, D2H)
            self.base = self.host.ctypes.data
        self.mode = mode
        self.reset_cache()

    def reset_cache(self):
        """Write back and forget every cached block."""
        if not hasattr(self, "slot_of"):
            return
        self.flush()
        if self.mode in ("managed", "managed_lazy"):
            for v in np.nonzero(self.block_in >= 0)[0]:
                prefetch(int(self.base) + int(self.block_in[v]) * self.block_bytes, self.block_bytes, LOC_HOST,
                         self.s_out)
            self.s_out.synchronize()
        self.slot_of[:] = -1
        self.block_in[:] = -1
        self.slot_key[:] = -1
        self.slot_key[self.K:] = np.iinfo(np.int64).max

    def set_cache_size(self, K):
        """Use only the first K cache slots (K <= the allocated size)."""
        assert K <= len(self.block_in)
        self.reset_cache()
        self.K = K
        self.chunk = max(1, min(8, K // 2))
        self.slot_key[K:] = np.iinfo(np.int64).max

    def _paging(self):
        return self.K > 0 and self.mode in ("paged", "managed", "managed_lazy")

    def _block_ptrs(self, blocks):
        return (np.uint64(self.base) + blocks.astype(np.uint64) * np.uint64(self.block_bytes)).astype(np.uint64)

    def _init_words(self, first_block=0, n_blocks=None, chunk=64):
        n_blocks = self.nb - first_block if n_blocks is None else n_blocks
        k = self.k["init_kernel"]
        with self.s_comp:
            if self.mode == "resident":
                nw = n_blocks * self.block_words
                off = first_block * self.block_words
                k(((nw + 255) // 256,), (256,), (self.dwords[off:], np.int64(nw), np.int64(off), np.uint32(self.seed)))
            else:
                tmp = cp.empty(chunk * self.block_words, cp.uint32)
                for b0 in range(first_block, first_block + n_blocks, chunk):
                    nbk = min(chunk, first_block + n_blocks - b0)
                    nw = nbk * self.block_words
                    k(((nw + 255) // 256,), (256,), (tmp, np.int64(nw), np.int64(b0 * self.block_words),
                                                     np.uint32(self.seed)))
                    self.host[b0 * self.block_words: b0 * self.block_words + nw] = cp.asnumpy(tmp[:nw])
                del tmp
        self.s_comp.synchronize()

    def words_of(self, blocks=None):
        """Host copy of the current words (for checks). Flushes the cache first."""
        self.flush()
        if self.mode == "resident":
            w = cp.asnumpy(self.dwords)
        elif self.mode in ("managed", "managed_lazy"):
            w = np.array(self.host)  # host access to managed memory migrates pages back to RAM
        else:
            w = self.host.copy()
        w = w.reshape(self.nb * self.bs, self.wpn)[: self.N]
        return w

    def load_words(self, words):
        """Overwrite all words from a host (N, B*S) uint32 array (for checks)."""
        flat = np.zeros(self.nb * self.block_words, np.uint32)
        flat[: self.N * self.wpn] = words.reshape(-1)
        if self.mode == "resident":
            self.dwords[:] = cp.asarray(flat)
        else:
            self.host[:] = flat
        if hasattr(self, "slot_of"):
            self.slot_of[:] = -1
            self.block_in[:] = -1
            self.slot_key[:] = -1
            self.slot_key[self.K:] = np.iinfo(np.int64).max

    def flush(self):
        """Write every dirty cached block back to host RAM (paged mode)."""
        self.s_comp.synchronize()
        if self.mode != "paged":
            return
        dirty = cp.asnumpy(self.slot_dirty)
        for v in np.nonzero((self.block_in >= 0) & (dirty > 0))[0]:
            b = int(self.block_in[v])
            cp.cuda.runtime.memcpyAsync(int(self.base) + b * self.block_bytes, int(self.slot_ptr[v]),
                                        self.block_bytes, D2H, self.s_out.ptr)
            self.d2h += self.block_bytes
        self.s_out.synchronize()
        with self.s_comp:
            self.slot_dirty.fill(0)
        self.s_comp.synchronize()

    # ------------------------------------------------------------------ paging
    def _run_blocks(self, blocks, launch, writes):
        """Make `blocks` addressable by the GPU and enqueue `launch` on them (on s_comp)."""
        if len(blocks) == 0:
            return
        if not self._paging():
            launch(self._block_ptrs(blocks), blocks, None)
            return
        self.op += 1
        base = self.op << 12
        events = {}
        if self.mode == "paged":
            dirty_h = cp.asnumpy(self.slot_dirty).astype(bool)  # state after the previous op
        slots = self.slot_of[blocks]
        hit = slots >= 0
        hb, hs = blocks[hit], slots[hit]
        if len(hb):
            launch(self._ptrs_for(hb, hs), hb, hs)
            events[0] = self.s_comp.record()
            self.slot_key[hs] = base
        mb = blocks[~hit]
        for ci, i in enumerate(range(0, len(mb), self.chunk), start=1):
            chunk = mb[i: i + self.chunk]
            n = len(chunk)
            vict = np.argpartition(self.slot_key, n - 1)[:n] if n < self.K else np.arange(self.K)
            vkeys = self.slot_key[vict]
            used_now = vkeys >= base
            if used_now.any():  # thrashing: wait until the compute that used these slots is done
                self.s_out.wait_event(events[int((vkeys[used_now] - base).max())])
            for v, un in zip(vict, used_now):
                ob = self.block_in[v]
                if ob < 0:
                    continue
                if self.mode == "paged":
                    if dirty_h[v] or (un and writes):
                        cp.cuda.runtime.memcpyAsync(int(self.base) + int(ob) * self.block_bytes,
                                                    int(self.slot_ptr[v]), self.block_bytes, D2H, self.s_out.ptr)
                        self.d2h += self.block_bytes
                else:
                    prefetch(int(self.base) + int(ob) * self.block_bytes, self.block_bytes, LOC_HOST, self.s_out)
                    self.d2h += self.block_bytes
                self.slot_of[ob] = -1
            self.s_in.wait_event(self.s_out.record())
            for b, v in zip(chunk, vict):
                if self.mode == "paged":
                    cp.cuda.runtime.memcpyAsync(int(self.slot_ptr[v]), int(self.base) + int(b) * self.block_bytes,
                                                self.block_bytes, H2D, self.s_in.ptr)
                elif self.mode == "managed":
                    prefetch(int(self.base) + int(b) * self.block_bytes, self.block_bytes, LOC_DEV, self.s_in)
                # managed_lazy: no hint; the kernel's page faults migrate the block (nominal bytes counted)
                self.h2d += self.block_bytes
                self.slot_of[b] = v
                self.block_in[v] = b
            if self.mode == "paged":
                dirty_h[vict] = False
                with self.s_in:
                    self.slot_dirty[cp.asarray(vict)] = 0
            self.s_comp.wait_event(self.s_in.record())
            launch(self._ptrs_for(chunk, vict), chunk, vict)
            events[ci] = self.s_comp.record()
            self.slot_key[vict] = base + ci

    def _ptrs_for(self, blocks, slots):
        if self.mode == "paged":
            return self.slot_ptr[slots]
        return self._block_ptrs(blocks)

    def _launch_args(self, ptrs, blks, slots):
        p = cp.asarray(ptrs.astype(np.uint64))
        b = cp.asarray(blks.astype(np.int32))
        s = cp.asarray(slots.astype(np.int32)) if slots is not None else b
        dirty = self.slot_dirty if (self.mode == "paged" and slots is not None) else None
        n = len(blks)
        grid = ((n * self.bs * 32 + 255) // 256,)
        return p, b, s, dirty, n, grid

    def _prep(self, blocks_dev, n):
        self.k["prep_kernel"](((n + 127) // 128,), (128,),
                              (blocks_dev, np.int32(n), self.flow_clock, self.fast_clock, np.int32(self.T),
                               self.mpow, np.float64(self.log_decay), self.bcoef))

    def _next_seed(self):
        self.seed = (self.seed * 1103515245 + 12345) & 0xFFFFFFFF
        return np.uint32(self.seed)

    # ------------------------------------------------------------------ input
    def _load_input(self, x, cols, learn_trace, p_in):
        k = self.k["input_kernel"]

        def run(c, xarr, row, mode, seed):
            if c is None or len(c) == 0:
                return
            cd = cp.asarray(c, cp.int32)
            n = len(c) * self.col_size
            k(((n + 255) // 256,), (256,), (cd, np.int32(len(c)), np.uint32(self.col_size), xarr, row,
                                            np.int32(mode), seed, np.float32(p_in)))

        if x is not None:  # dense input (baseline-style)
            self.x[:] = cp.asarray(x, cp.float32)
            self.x_cols = None
            if learn_trace:
                self.pre[self.ring] = self.x.astype(cp.float16)
                self.pre_cols[self.ring] = None
        else:  # structured, sparse input: only the listed columns are on
            seed = self._next_seed()
            if self.x_cols is not None:
                run(self.x_cols, self.x, None, 0, seed)
            if learn_trace:
                old = self.pre_cols[self.ring]
                if old is not None:
                    run(old, None, self.pre[self.ring], 0, seed)
                run(cols, self.x, self.pre[self.ring], 1, seed)
                self.pre_cols[self.ring] = np.asarray(cols)
            else:
                run(cols, self.x, None, 1, seed)
            self.x_cols = np.asarray(cols)

    # ------------------------------------------------------------------ acting
    def forward(self, x=None, context=0.0, learn_trace=True, active=None, cols=None, p_in=0.2):
        """Dense x (like PackedPopulation.forward), or structured input given as active
        columns `cols` plus the blocks they touch (`active`). Returns the (N,) rate array."""
        t = self.t
        with self.s_comp:
            if learn_trace:
                self.ring = (self.ring + 1) & 3
            self._load_input(x, cols, learn_trace, p_in)
            blocks = np.arange(self.nb, dtype=np.int64) if active is None else np.asarray(active, np.int64)
            bd = cp.asarray(blocks.astype(np.int32))
            self._prep(bd, len(blocks))
            ctx_mult = np.float32(1.0 + t.context_gain * context)
            kf = self.k["forward_kernel"]

            def launch(ptrs, blks, slots):
                p, b, s, dirty, n, grid = self._launch_args(ptrs, blks, slots)
                kf(grid, (256,), (p, b, s, np.int32(n), np.int32(self.bs), np.int64(self.N), self.bcoef, self.ids,
                                  self.gate, self.thr, self.post, np.int32(self.ring), self.x, np.uint32(self.npc),
                                  np.uint32(self.col_size), np.uint32(self.col_mask), ctx_mult,
                                  np.int32(bool(learn_trace)), self.rate, dirty, self.counters))

            self._run_blocks(blocks, launch, writes=learn_trace)
            if len(blocks) < self.nb:
                self.blk_active.fill(0)
                self.blk_active[bd] = 1
                self.k["quiet_kernel"](((self.N + 255) // 256,), (256,),
                                       (self.blk_active, np.int32(self.bs), np.int64(self.N), self.thr, self.post,
                                        np.int32(self.ring), np.int32(bool(learn_trace)), self.rate))
        if learn_trace:
            self.fwd_count += 1
            self.last_active[blocks] = self.fwd_count
        self.s_comp.synchronize()  # public ops are synchronous: results are ready on return
        return self.rate

    # ------------------------------------------------------------------ learning
    def recent_blocks(self):
        """Blocks with any input in the last ELIG_STEPS steps: the only ones with eligibility."""
        return np.nonzero(self.last_active > self.fwd_count - pn.ELIG_STEPS)[0]

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0, blocks=None):
        if dopamine == 0.0:
            return
        t = self.t
        blocks = self.recent_blocks() if blocks is None else np.asarray(blocks, np.int64)
        if len(blocks) == 0:
            return
        oneshot = abs(dopamine) * ach > t.oneshot_level
        km = self.k["modulate_kernel"]
        with self.s_comp:
            bd = cp.asarray(blocks.astype(np.int32))
            self._prep(bd, len(blocks))
            seed = self._next_seed()
            ach_h, dop_h = np.float32(np.float16(ach)), np.float32(np.float16(dopamine))
            tag_gain = np.float32(arousal * abs(dopamine))

            def launch(ptrs, blks, slots):
                p, b, s, dirty, n, grid = self._launch_args(ptrs, blks, slots)
                km(grid, (256,), (p, b, s, np.int32(n), np.int32(self.bs), np.int64(self.N), self.bcoef, self.ids,
                                  self.gate, self.plast, self.post, self.pre, np.int64(self.n_inputs),
                                  np.int32(self.ring), np.uint32(self.npc), np.uint32(self.col_size),
                                  np.uint32(self.col_mask), ach_h, dop_h, tag_gain, np.int32(oneshot), seed,
                                  dirty, self.counters))

            self._run_blocks(blocks, launch, writes=True)
            self.fast_clock[bd] = self.T
            if oneshot:
                self.flow_clock[bd] = self.T
        self.s_comp.synchronize()

    def tick(self):
        self.T += 1  # everything else is lazy (per-block clocks)

    def rest(self, steps):
        self.T += int(steps)

    # ------------------------------------------------------------------ sleeping
    def sleep(self, blocks=None):
        t = self.t
        blocks = np.arange(self.nb, dtype=np.int64) if blocks is None else blocks
        ks = self.k["sleep_kernel"]
        with self.s_comp:
            before = int(self.counters[1].get())
            bd = cp.asarray(blocks.astype(np.int32))
            self._prep(bd, len(blocks))
            seed = self._next_seed()

            def launch(ptrs, blks, slots):
                p, b, s, dirty, n, grid = self._launch_args(ptrs, blks, slots)
                ks(grid, (256,), (p, b, s, np.int32(n), np.int32(self.bs), np.int64(self.N), self.bcoef,
                                  *[np.float32(v) for v in self.sleep_m], seed, dirty, self.counters))

            self._run_blocks(blocks, launch, writes=True)
            self.flow_clock[bd] = self.T
            self.fast_clock[bd] = self.T
            self.post.fill(0)
            self.pre.fill(0)
            self.pre_cols = [None] * 4
            self.age += 1
            # numpy does this in float16 (weak python scalars)
            fl, mat = np.float16(t.plasticity_floor), np.float16(t.maturation)
            a = (self.plast.astype(cp.float32) - np.float32(fl)).astype(cp.float16)
            a = (a.astype(cp.float32) * np.float32(mat)).astype(cp.float16)
            self.plast = (a.astype(cp.float32) + np.float32(fl)).astype(cp.float16)
            pruned = int(self.counters[1].get()) - before
        self.s_comp.synchronize()
        self.last_active[:] = -(1 << 40)
        return pruned

    # ------------------------------------------------------------------ sizes
    def vram_state_bytes(self):
        per = [self.ids, self.gate, self.thr, self.plast, self.post, self.age, self.rate]
        blk = [self.flow_clock, self.fast_clock, self.bcoef, self.blk_active]
        shared = [self.x, self.pre]
        words = (self.dwords.nbytes if self.mode == "resident" else
                 self.K * self.block_bytes if self.mode == "paged" else 0)
        return {"per_neuron": sum(a.nbytes for a in per) / self.N,
                "per_block": sum(a.nbytes for a in blk) / self.N,
                "shared_inputs": sum(a.nbytes for a in shared) / self.N,
                "words": words / self.N}

    def lines_written(self):
        return int(self.counters[2].get())

    def host_bytes(self):
        return 0 if self.host is None else self.host.nbytes

    def sync(self):
        self.s_comp.synchronize()
        self.s_in.synchronize()
        self.s_out.synchronize()
