"""
A LittleBrain character: a federated brain with a basal ganglia action loop (brainspec.stage_bg) playing a
game (prototype/games). Each turn:

  1. look:    the picture is shown for `look` steps; the brain's GPi output per action channel is read over the
              last 3 steps. Lower GPi = that action is released (disinhibited) more.
  2. choose:  softmax over the channels' release (temperature = how much neural noise drives exploration).
  3. act:     the efference copy of the chosen action excites its striatal channel for `act` steps, the game
              answers, and dopamine (reward minus expected reward) arrives while the eligibility traces are hot.
  4. rest:    a short blank.

Dopamine = reward - expected reward. critic="average": one running average of reward. critic="cortex": the
expectation is read out of the cortical activity during the look phase (a linear readout learned by the delta
rule), a stand-in for striosomes / ventral striatum, which learn what each situation is worth (notes/30).
"""

import os

import numpy as np

from backend import to_host, xpof
from brainspec import action_channel


class BrainAgent:
    def __init__(self, brain, n_actions, look=20, read=16, act=4, rest=3, temperature=0.02, ach=0.6, seed=0,
                 learn=True, dopamine_gain=1.0, critic="average", critic_lr=0.3):
        self.b, self.n = brain, n_actions
        self.look, self.act, self.rest, self.read = look, act, rest, read   # read: last steps of the look that count
        self.temperature, self.ach, self.learn, self.dopamine_gain = temperature, ach, learn, dopamine_gain
        self.rng = np.random.default_rng(seed)
        self.timed = getattr(brain.spec, "timed_dopamine", False)
        self.cue_value = 0.0
        self.body_signals = {}
        self.expected = 0.0
        self.critic, self.critic_lr, self.w_value, self.features = critic, critic_lr, None, None
        self.value_groups = ("A.L23", "A.L5IT", "A.L5PT")
        self.chan = {k: action_channel(to_host(brain.tile[k]), n_actions) for k in ("MOT", "GPi", "STR.D1", "STR.D2")}
        P = brain.pos["retina"]
        self._px = P
        # A brain with a retina (vision.py) gets a real eye; older brains sample the picture as before
        self.eye = None
        if "RET.on" in brain.pos:
            from retina import Retina
            self.eye = Retina(brain)
        # Hearing (note 81): sound or text in an observation reaches the ear one frame per brain step
        self.ear, self.hearing = None, None
        if "EAR.sus" in brain.pos:
            from cochlea import Ear
            self.ear = Ear(brain)
        # Knowing where you are (note 83): self-motion reported by the game drives head-direction and grid cells
        self.place = None
        if "MEC.grid" in brain.pos or "ATN.hd" in brain.pos:
            from spatial import SelfMotion
            self.place = SelfMotion(brain)
        # A world model and look-ahead (WORLD_MODEL=1, note 96): learned on the place code, it imagines each action's
        # outcome before the choice and biases the basal ganglia
        self.wm, self.wm_prev, self.wm_phi, self.plan_bonus = None, None, None, None
        if os.environ.get("WORLD_MODEL", "0") == "1" and "MEC.grid" in brain.pos:
            from world_model import WorldModel
            self.wm = WorldModel(100, n_actions, seed=seed)
            self.wm_beta = float(os.environ.get("WM_BETA", "10"))   # strong enough to override the habit (note 96)
        # Eye movements (note 76): with a superior colliculus, the eye fixates the centre at the start of each look,
        # and at `saccade_at` steps jumps to where the colliculus points (VISION_SACCADES=0: eye held still)
        self.saccades = self.eye is not None and "SC.deep" in brain.g and os.environ.get("VISION_SACCADES", "1") == "1"
        self.saccade_at, self.gaze_log = 6, []
        # Credit across steps (plan phase 1, note 80): with TD_GAMMA > 0 the value of each situation is learned by
        # temporal-difference errors, and the dopamine at the next turn's cue (gamma x its value) also credits the
        # previous action. Opt-in; 0 keeps the one-turn behaviour.
        self.td_gamma = float(os.environ.get("TD_GAMMA", getattr(brain, "td_gamma", 0.0) or 0.0))
        self.td_prev, self.v_now, self.cue_dopamine, self.episode_over = None, 0.0, 0.0, False
        self.da_rest = None   # dopamine's resting level: a slow average of its reading over all steps
        # TD(lambda): earlier turns of an episode keep fading eligibility (gamma*lambda per turn), so a chain's first
        # steps get credit sooner (note 80)
        self.td_lambda = float(os.environ.get("TD_LAMBDA", "0.6"))
        self.td_hist = []     # [sensory trace, motor copy, value code, eligibility] of earlier turns, newest first
        self.last_info = {}
        self.last = {}
        # Dopamine from the brain's own dopamine cells if it has them (reward goes in through the US input);
        # otherwise the stand-in critic below computes it
        self.own_dopamine = "US" in brain.pos and "VTA.DA" in brain.g
        # Called after every brain step as on_step(phase, info): phase is "look", "act" or "rest". The visualizer
        # uses it to draw (and to pause the brain between steps).
        self.on_step = None
        self.turn = self.turns_played = 0
        self.pending = None
        self.game = self.next_observation = None
        self.sleep_requested = False

    def _tick(self, phase, **info):
        if self.on_step is not None:
            self.on_step(phase, info)

    def _retina(self, image):
        if self.eye is not None:
            from retina import Sight
            return Sight(self.eye, image)
        image = np.asarray(image)
        if image.ndim == 3:   # colour picture, older brain: brightness only
            image = image @ np.array([0.299, 0.587, 0.114], np.float32)
        h, w = image.shape
        ix = np.minimum((self._px[:, 0] * w).astype(int), w - 1)
        iy = np.minimum((self._px[:, 1] * h).astype(int), h - 1)
        return image[iy, ix].astype(np.float32)

    def _input_scale(self, like_):
        """What the striatum's centred input is divided by: its running size (note 77), or each input's own running
        spread (STRIATUM_NORM=feature, note 89)."""
        var = getattr(self.b, "trace_var", None)
        if os.environ.get("STRIATUM_NORM", "size") == "feature" and var is not None and var.shape == like_.shape:
            xp = xpof(like_)
            return xp.sqrt(var + 1e-6) * np.sqrt(float(var.size))
        return getattr(self.b, "trace_scale", None) or 1.0

    def _inputs(self, x, action=None, us=0.0):
        d = self._inputs_seen(x, action, us)
        if self.ear is not None:
            import cochlea
            d.update(self.hearing.channels() if self.hearing is not None else self.ear(np.zeros(cochlea.BANDS, np.float32)))
        if self.place is not None and not ({"MEC", "ATN"} & set(self.b.ablated)):
            d.update(self.place.channels())
        return d

    def _inputs_seen(self, x, action=None, us=0.0):
        if self.eye is not None:
            quiet = getattr(self, "_quiet", None)
            if quiet is None:
                quiet = self._quiet = {k: np.zeros(len(self.b.pos[k]), np.float32) for k in self.b.spec.inputs}
            d = {**quiet, **x.channels(), "MOT": self._mot(action)}
        else:
            d = {"retina": x, "MOT": self._mot(action)}
        if self.own_dopamine:
            d["US"] = np.full(len(self.b.pos["US"]), us, np.float32)
        return d

    def _teach(self, delta):
        """The temporal-difference error as a teaching signal (note 80). The critic computes it and it drives the
        dopamine cells (bursts directly, pauses through the lateral habenula); a negative error teaches only if the
        habenula -> RMTg pathway is intact, so silencing it removes learning about harm (track N1)."""
        delta = float(np.clip(delta, -1.5, 1.5))
        if delta < 0 and ({"LHb", "LHb.proj", "RMTg", "RMTg.GABA"} & set(self.b.ablated)):
            return 0.0
        return delta

    def _rest_update(self):
        d = float(self.b.last_dopamine)
        self.da_rest = d if self.da_rest is None else self.da_rest + 0.01 * (d - self.da_rest)

    def _raw(self, avg, update=False):
        """The striatum's sensory input for the value critic: minus a very slow average (~200 turns, so the part every
        picture shares is removed but a frequent situation keeps its value), at a steady size (note 80)."""
        m = getattr(self.b, "value_mean", None)
        if m is None or m.shape != avg.shape:
            m = avg * 0
        dev = avg - m
        n = float(np.linalg.norm(to_host(dev)))
        size = getattr(self.b, "raw_scale", None)
        if update:
            self.b.value_mean = m + 0.005 * (avg - m)
            size = n if not size else size + 0.05 * (n - size)
            self.b.raw_scale = max(size, 1e-3)
        x = dev / (getattr(self.b, "raw_scale", None) or max(n, 1e-3))
        # sparse code: only the most active features (top 5%), all-or-none, so different situations barely overlap
        # and learning one situation's value doesn't disturb another's (pattern separation, as in the dentate gyrus)
        xs = to_host(x)
        k = max(1, int(0.05 * xs.size))
        code = np.zeros_like(xs)
        code[np.argpartition(-xs, k)[:k]] = 1.0 / np.sqrt(k)
        return code

    def _centred(self, avg):
        """The striatum's sensory input as learning sees it: minus recent turns, at a steady size (notes 74, 77)."""
        m = getattr(self.b, "trace_mean", None)
        size = getattr(self.b, "trace_scale", None) or 1.0
        return (avg - m) / size if m is not None and m.shape == avg.shape else avg

    def _mot(self, action=None):
        m = np.zeros(len(self.chan["MOT"]), np.float32)
        if action is not None:
            m[self.chan["MOT"] == action] = 1.0
        return m

    def _per_channel(self, key):
        r = to_host(self.b.g[key].rate)
        return np.array([r[self.chan[key] == a].mean() for a in range(self.n)])

    def choose(self, image):
        moved = False
        if self.eye is not None:
            self.eye.look_at((0.5, 0.5))   # fixate the centre
        x, gpi, feats = self._retina(image), [], []
        sensed = []   # vision brains: the striatum's sensory input integrated over the read window (note 74)
        td = self.td_gamma > 0 and self.timed and "TH.core" in self.b.g
        cue_at = self.look - 4   # the value signal arrives a few steps before the choice; dopamine is read after it
        cue_da, pre_da = 0.0, 0.0
        for s in range(self.look):
            signals = dict(self.body_signals)
            if td and s == cue_at:
                from striatal_projection import activity, critic_value
                feat = self._raw(sum(sensed) / len(sensed), update=True) if sensed else activity(self.b)
                self.value_trace = feat     # the critic learns on exactly the code it predicted from
                self.v_now = critic_value(self.b, feat)
                cv = self.td_gamma * self.v_now
                self.cue_value = self.v_now
                if "LHb.proj" in self.b.g and any(p.id == "aversive value -> LHb" for p in self.b.spec.pooled):
                    # better than nothing excites dopamine cells; worse inhibits them through the habenula (N1)
                    signals["value_change"], signals["value_loss"] = max(cv, 0.0), max(-cv, 0.0)
                else:
                    signals["value_change"] = cv
                if self.learn and self.td_prev is not None:
                    prev_mot = self.td_prev[1]
                    tiles = to_host(self.b.tile["MOT"])
                    signals["credit_motor"] = (np.bincount(tiles, weights=prev_mot, minlength=64)
                                               / np.maximum(np.bincount(tiles, minlength=64), 1)).astype(np.float32)
            elif self.timed and not td and s == self.look - 1:
                rates = self.b.g
                value = (float(to_host(rates["STR.SO"].rate).mean())
                         + 4 * float(to_host(rates["BLA.pos"].rate - rates["BLA.neg"].rate).mean()))
                if "TH.core" in rates:
                    # Sample the BLA projection ensembles before their unconditioned
                    # input arrives. Striosomes supply a smaller positive expectation.
                    appetitive = rates["BLA.pos"].rate.reshape(64, -1)[:, 0].mean()
                    aversive = rates["BLA.neg"].rate.reshape(64, -1)[:, 0].mean()
                    value = float(to_host(appetitive - aversive + .15 * rates["STR.SO"].rate.mean()))
                self.cue_value = float(np.clip(value, -1, 1))
                signals["value_change"] = self.cue_value
            self.b.step(self._inputs(x), learn=self.learn, ach=self.ach, signals=signals)
            if td and s >= cue_at:
                cue_da += self.b.last_dopamine - self.da_rest   # against dopamine's resting level
            self._rest_update()
            if self.saccades and s >= 1 and not moved and s <= self.saccade_at:
                from vision import saccade_target
                target = saccade_target(self.b, self.eye)
                if target is not None:   # the colliculus bursts: look there (once per look)
                    self.eye.look_at(target)
                    moved = True
                if moved or s == self.saccade_at:
                    self.gaze_log.append(tuple(self.eye.gaze))
            self._tick("look", image=image, step=s, gaze=tuple(self.eye.gaze) if self.eye is not None else None)
            if s >= self.look - self.read:
                gpi.append(self._per_channel("GPi"))
                feats.append(np.concatenate([to_host(self.b.g[k].rate) for k in self.value_groups]))
                if self.eye is not None and "TH.core" in self.b.g:
                    from striatal_projection import activity
                    sensed.append(activity(self.b))
                    # the striatum's drive for this choice is the same integrated, centred input learning uses
                    m = getattr(self.b, "trace_mean", None)
                    avg = sum(sensed) / len(sensed)
                    size = self._input_scale(avg)
                    self.b.cue_online = (avg - m) / size if m is not None and m.shape == avg.shape else None
        if "TH.core" in self.b.g:
            from striatal_projection import activity
            # Eligibility traces integrate presynaptic activity over about a second; a single step of cortical
            # activity is too noisy (vision brains only, so older results stay reproducible)
            self.sensory_trace = sum(sensed) / len(sensed) if sensed else activity(self.b)
            self.b.cue_online = None
            if sensed:
                # What is different about this picture from the pictures usually seen: subtract a running average of
                # recent turns' inputs (habituation at the trial timescale). Without it, "a picture is on" (shared by
                # every picture) dominated learning and colour was never learned (note 74).
                m = getattr(self.b, "trace_mean", None)
                tr = self.sensory_trace
                m = tr * 0 if m is None or m.shape != tr.shape else m
                # ...rescaled to a steady typical size (a slow running average of its size): after centring the
                # input was ~0.1 in size, so its learned drive was too weak to steer the choice (note 77)
                dev = tr - m
                size = getattr(self.b, "trace_scale", None)
                n = float(np.linalg.norm(to_host(dev)))
                size = n if not size else size + 0.05 * (n - size)
                self.b.trace_scale = max(size, 1e-3)
                if os.environ.get("STRIATUM_NORM", "size") == "feature":
                    # each input by its own running spread (note 89): an informative but quiet input counts as much as
                    # a loud one, as input-specific gain control would make it; the whole vector keeps size ~1
                    var = getattr(self.b, "trace_var", None)
                    var = dev * dev if var is None or var.shape != dev.shape else var + 0.02 * (dev * dev - var)
                    self.b.trace_var = var
                self.sensory_trace = dev / self._input_scale(dev)
                self.b.trace_mean = m + 0.05 * (tr - m)
                if not td:
                    self.value_trace = None
            if td and self.learn and self.td_prev is not None:
                # gamma x the value of where the last action led completes the previous turn's temporal-difference
                # error (its outcome part was applied at its outcome): credit the previous action and situation
                from striatal_projection import critic_learn, reinforce
                delta = self._teach(self.td_gamma * self.v_now)
                for tr_, mot_, vt_, e_ in self.td_hist:       # the previous turn (eligibility 1) and fading earlier ones
                    reinforce(self.b, tr_, mot_, delta * e_)
                    critic_learn(self.b, vt_, delta * e_)
            self.cue_dopamine = cue_da
            self.timing_trace = None
            if hasattr(self.b, "parallel_fiber_ltd"):
                from association import timing_activity
                self.timing_trace = timing_activity(self.b)
        self.features = np.mean(feats, 0)
        release = -np.mean(gpi, 0)
        if self.plan_bonus is not None and len(self.plan_bonus) == len(release):
            release = release + self.plan_bonus   # the imagined outcomes bias the choice (note 96)
        p = np.exp((release - release.max()) / self.temperature)
        p /= p.sum()
        action = int(self.rng.choice(self.n, p=p))
        self.last = {"release": release, "p": p, "D1": self._per_channel("STR.D1"), "D2": self._per_channel("STR.D2")}
        return action, x

    def outcome(self, x, action, reward):
        self.b.last_reward = float(reward)   # for the striatal territories' trust (note 93)
        if self.critic == "cortex":
            f = self.features
            if self.w_value is None:
                self.w_value = np.zeros_like(f)
            expected = float(self.expected + self.w_value @ f)
            delta = reward - expected
            if self.learn:  # normalized delta rule; the bias term is the running average
                self.w_value += self.critic_lr * delta * f / (f @ f + 1.0)
                self.expected += 0.05 * delta
        else:
            expected = self.expected
            self.expected += 0.05 * (reward - self.expected)
        rpe = float(np.clip(reward - expected, -1, 1)) * self.dopamine_gain
        if self.own_dopamine:
            feedback = self.last_info.get("feedback", True)
            rpe = 0.0
            # credit across steps: the outcome's dopamine is read relative to just before it, over a window that
            # covers the habenula's delayed dip (act steps 2-3 and the first two rest steps), then learned once
            tdw = feedback and self.td_gamma > 0 and self.timed and "TH.core" in self.b.g
            pre_da, out_da = 0.0, 0.0
            for s in range(self.act):
                due = feedback and s == 2
                signals = dict(self.body_signals, outcome_due=due,
                               timing_error=self.last_info.get("timing_error", 0.0) if due else 0.0,
                               value_change=(-self.cue_value if self.timed and due else 0.0))
                if due and self.td_gamma > 0 and self.timed and "TH.core" in self.b.g:
                    # the outcome is measured against the situation's full value (not gamma-discounted)
                    v = self.v_now
                    if "LHb.proj" in self.b.g and any(p.id == "aversive value -> LHb" for p in self.b.spec.pooled):
                        signals["value_change"], signals["value_loss"] = max(-v, 0.0), max(v, 0.0)
                    else:
                        signals["value_change"] = -v
                if due and "timing_offset" in self.last_info:
                    signals["timing_offset"] = self.last_info["timing_offset"]
                self.b.step(self._inputs(x, action, us=reward if due else 0.0),
                            learn=self.learn, ach=self.ach, arousal=0.3, signals=signals)
                if tdw and s >= 2:
                    out_da += self.b.last_dopamine - self.da_rest
                self._rest_update()
                if due and tdw:
                    pass   # with credit across steps, outcome learning waits for the habenula's delayed dip (below)
                elif due:
                    rpe = self.b.last_dopamine
                    if self.learn and "TH.core" in self.b.g:
                        from striatal_projection import food_outcome, pair_outcome, reinforce
                        if "food_contact" in self.last_info:
                            amount = self.last_info.get("food_reward", 1.) if self.last_info["food_contact"] else 0.
                            food_outcome(self.b, self.sensory_trace, self._mot(action), amount)
                        if "timing_error" in self.last_info:
                            from association import climbing_fiber_credit
                            climbing_fiber_credit(self.b, self.timing_trace, self._mot(action))
                        pair_outcome(self.b, self.sensory_trace, reward)
                        reinforce(self.b, self.sensory_trace, self._mot(action), rpe)
                    if self.learn and self.timed:
                        self.b.record_waking_outcome(rpe)
                self._tick("act", action=action, reward=reward, dopamine=self.b.last_dopamine, step=s)
        else:
            for s in range(self.act):
                self.b.step(self._inputs(x, action), learn=self.learn, ach=self.ach,
                            dopamine=rpe if (self.learn and s == 2) else 0.0, arousal=0.3)
                self._tick("act", action=action, reward=reward, dopamine=rpe if s == 2 else 0.0, step=s)
        for s in range(self.rest):
            self.b.step(self._inputs(0 * x), ach=self.ach)
            if self.own_dopamine and tdw and s < 2:
                out_da += self.b.last_dopamine - self.da_rest
            self._rest_update()
            self._tick("rest", step=s)
        if self.own_dopamine and tdw:
            rpe = self._teach(float(reward) - self.v_now)   # reward minus this situation's value
            if self.learn:
                from striatal_projection import critic_learn, food_outcome, pair_outcome, reinforce
                if "food_contact" in self.last_info:
                    amount = self.last_info.get("food_reward", 1.) if self.last_info["food_contact"] else 0.
                    food_outcome(self.b, self.sensory_trace, self._mot(action), amount)
                if "timing_error" in self.last_info:
                    from association import climbing_fiber_credit
                    climbing_fiber_credit(self.b, self.timing_trace, self._mot(action))
                pair_outcome(self.b, self.sensory_trace, reward)
                reinforce(self.b, self.sensory_trace, self._mot(action), rpe)
                critic_learn(self.b, self.value_trace if getattr(self, "value_trace", None) is not None
                             else self.sensory_trace, rpe)
                for tr_, mot_, vt_, e_ in self.td_hist:       # earlier turns of the episode, fading
                    w = e_ * self.td_gamma * self.td_lambda
                    reinforce(self.b, tr_, mot_, rpe * w)
                    critic_learn(self.b, vt_, rpe * w)
                if self.timed:
                    self.b.record_waking_outcome(rpe)
        return rpe

    def begin_turn(self, obs):
        if self.pending is not None:
            raise ValueError("the previous action still needs its outcome")
        self.turn = self.turns_played
        self.body_signals = {k: float(obs.body[k]) for k in ("thirst", "social_contact") if k in obs.body}
        if "hunger" in obs.body:
            hunger = float(obs.body["hunger"])
            self.body_signals.update(hunger=hunger, satiety=1.0 - hunger)
        frames = getattr(obs, "frames", None)
        if self.place is not None:
            self.place.move(getattr(obs, "motion", None), obs.body)
        self.plan_bonus = None
        if self.wm is not None:
            from world_model import HIPPOCAMPAL
            phi = self.place.place_code()
            if self.wm_prev is not None and self.learn:
                p_, a_, r_ = self.wm_prev
                self.wm.observe(p_, a_, r_, phi)
            self.wm_prev, self.wm_phi = None, phi
            if not (HIPPOCAMPAL & set(self.b.ablated)):
                q = self.wm.q(phi)
                self.plan_bonus = self.wm_beta * (q - q.mean())
        self.hearing = None
        if self.ear is not None:
            import cochlea
            sound = getattr(obs, "sound", None)
            if sound is None and getattr(obs, "text", None):
                sound = cochlea.text_sound(obs.text)
            if sound is not None:
                self.hearing = cochlea.Hearing(self.ear, sound)
        action, retina = self.choose(frames if (frames and self.eye is not None) else obs.image)
        self.pending = action, retina
        return action

    def finish_turn(self, reward, info=None, sleep_every=None):
        if self.pending is None:
            raise ValueError("observe before reporting an outcome")
        action, retina = self.pending
        self.last_info = dict(info or {})
        food = self.last_info.get("food_reward", 0.0)
        if self.last_info.get("food_contact") and "HYP.AgRP" in self.b.g:
            # Pre-meal drive modulates only the food component of the outcome.
            balance = float(to_host(self.b.g["HYP.AgRP"].rate - self.b.g["HYP.POMC"].rate).mean())
            reward = np.clip(reward - food + food * balance, -1.0, 1.0)
        waiting = self.last_info.get("action_applied") is False
        if "TH.core" in self.b.g and waiting:
            # An unapplied motor command leaves the observed context in working memory.
            self.b.hold_context = True
        self.last_rpe = self.outcome(retina, action, reward)
        if self.wm is not None and self.wm_phi is not None and self.learn:
            if self.episode_over or bool(self.last_info.get("done", False)):
                self.wm.observe(self.wm_phi, action, float(reward), None)
            else:
                self.wm_prev = (self.wm_phi, action, float(reward))
        # remember this turn for the next cue's temporal-difference credit, unless the episode ended here
        done = self.episode_over or bool(self.last_info.get("done", False))
        trace = getattr(self, "sensory_trace", None)
        vt = getattr(self, "value_trace", None)
        self.td_prev = None if (done or trace is None) else (trace, self._mot(action), vt if vt is not None else trace)
        if self.td_gamma > 0:
            if done or trace is None:
                self.td_hist = []
            else:
                decay = self.td_gamma * self.td_lambda
                self.td_hist = [[a, m, v, e * decay] for a, m, v, e in self.td_hist if e * decay > 0.05]
                self.td_hist.insert(0, [trace, self._mot(action), vt if vt is not None else trace, 1.0])
        self.episode_over = False
        if getattr(self.b, "hold_context", False) and not waiting:
            self.b.hold_context = False
            self.b.clear_context_steps = 5
        self.turns_played += 1
        if sleep_every is None:
            sleep_every = 64 if "MS.theta" in self.b.g else 0
        if self.learn and (self.sleep_requested or (sleep_every and self.turns_played % sleep_every == 0)):
            self.b.replay_and_sleep()
            self.sleep_requested = False
        self.pending = None
        return self.last_rpe

    def play(self, game, turns, sleep_every=None, log=None):
        """Continue the current episode across calls, including single-turn viewer updates."""
        if game is not self.game:
            self.game, self.next_observation = game, game.reset()
        rewards = []
        for _ in range(turns):
            action = self.begin_turn(self.next_observation)
            obs, reward, done, info = game.step(action)
            self.episode_over = bool(done)
            self.finish_turn(reward, info, sleep_every)
            rewards.append(reward)
            if log is not None:
                log.append({"turn": self.turn, "action": action, "reward": reward,
                            "dopamine": self.last_rpe, **info, **self.last})
            self.next_observation = game.reset() if done else obs
        return rewards
