## Watch a brain make a choice

A picture flashes. Four buttons light up. One button earns a reward. LittleBrain is an open-source simulator where you can watch a character learn which one to press. Pause on a surprising choice, hover over a neuron, and compare what the character saw with what happened next.

## How the neurons were designed

The reference brain contains **18,944 simulated neurons drawn from 98 modeled human cell types**. A cortical pyramidal cell gathers signals along branches called dendrites. Nearby inhibitory cells can quiet the cell or dampen input to particular branches. The simulation simplifies each cell, while preserving differences in where signals arrive, which cells they reach, and what makes those connections change. On the map you can follow a visual cue from the retina into the brain and watch the circuits involved in choosing an action.

## How experience changes a connection

When a picture appears and the character presses a button, recently active connections keep a brief trace. If the outcome is better or worse than expected, simulated dopamine helps strengthen or weaken eligible connections. The next time the same picture appears, the odds of choosing that button can shift. During sleep, the brain replays recent activity and a small share of synapses can change partners.

## What makes LittleBrain different

Many AI models learn by calculating a prediction error and sending it backward through layers. LittleBrain learns during play through local changes at individual connections. You can see the cells responding to a cue alongside the choice the character makes. Switch learning off and replay the signal game to see how feedback changes the same brain’s behavior.
