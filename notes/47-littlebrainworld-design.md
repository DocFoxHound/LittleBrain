# 47. LittleBrainWorld design

LittleBrainWorld would be a small top-down world where a brain can learn what happens when it moves, touches something, or acts near another character. Its first room should answer one question we already know how to ask through the Game Bridge. Can a brain use a visible cue to choose a direction? Once that works, the same world gives us a controlled place to introduce one new sense at a time.

## The first room

Start with one character at a fork in a small tile grid. It sees a bright patch down one of two passages. A left or right step enters the corresponding passage and ends the trial. Entering the lit passage gives a positive outcome. Entering the other gives zero. Balance the two sides and randomize their order. Reset the room between trials and keep the same brain alive so its experience carries over.

The character receives a small grayscale image centered on its body. Walls block its view. Shapes and brightness carry the cue. A human watching can see the whole map, the character's exact view, its choice and the outcome side by side. The task definition and the spectator overlay stay outside the character's inputs. A second map can move the fork or change the passage lengths to test what carries over.

## What the world needs

| Part | Responsibility |
|---|---|
| World simulation | Own the map, body positions, collision rules and seeded randomness. Advance by a fixed number of ticks per decision. |
| Character interface | Produce the local image and available body readings. Apply actions to the body. A scripted controller and a brain must be able to drive the same character. |
| LittleBrain adapter | Run one persistent Game Bridge process per character, using the default full brain. Send observations, apply the chosen movement once, and acknowledge its consequence. |
| Trial runner | Choose maps and starting conditions, schedule trials, assign outcomes and score choices. Store hidden task state in the experiment log. |
| Headless runner and viewer | Run the same simulation and character camera on a cloud machine or a desktop. Opening, slowing or closing the spectator window should have no effect on a trial. |

For example, suppose the patch is on the left. The adapter sends the local image, receives the action `left`, moves the character one tile and reports the positive outcome. It waits for the bridge's `applied` reply, which includes completion of any scheduled sleep, before starting the next trial. Wall-clock delays advance zero world ticks. The existing bridge supports the whole exchange with two named actions. [Game Bridge protocol](45-game-bridge.md)

## Repetitions and the first acceptance test

Headless runs still generate the pixels the retina needs. They can skip drawing the spectator view and run the next decision as soon as the previous one completes. Record the game and brain revisions, configuration, seeds, starting checkpoint identity, actions, observation checksums, outcomes and episode boundaries. Replaying recorded actions from the same world seed should reproduce the images and outcomes with the viewer open or closed. A saved brain alone cannot reconstruct a running world.

1. Drive the fork with a scripted controller that inspects only the pixels. Require every lit passage to be reachable and correctly scored, and an always-left controller to score exactly 50% over a balanced block. Verify replay before connecting a brain.
2. For brain seeds 0, 1 and 2, train one copy of each starting brain for 300 choices. Run a matched copy with learning disabled on the same cue schedule. Freeze learning for evaluation and give both copies the same 100 fresh, balanced trials. A proposed gate is at least 70 correct choices for each trained seed and a higher score than its matched control. Report each pair. If a seed misses, inspect the image, action and outcome for that trial before adding another input.
3. Measure completed **brain decisions per wall-clock second** and elapsed time per 100 trials, including scheduled sleep. Record brain startup separately. Benchmark one world first, then try independent worlds in parallel and report aggregate throughput and peak memory. These are proposed measurements.

## Adding senses and controls

| Stage | What the character experiences | How to tell whether it matters |
|---|---|---|
| Touch | A step into a wall produces a brief contact at the front of the body. Deliver it with the next observation, even if contact has already ended. Later, the same body can register sustained contact, immersion, heat and injury separately. | Train in darkness with balanced blocked and clear trials. Give the same initial forward step and neutral outcome in both. Score the next binary choice, turn if blocked or advance if clear. Hold other cues constant. Compare frozen performance with intact and shuffled touch. Chance is 50%. |
| Body state and movement | Food contact, hunger and thirst can use existing bridge fields. A future movement sense can report actual displacement when a step slips or hits a wall. | Offer the same food choices when fed and hungry. In a separate movement test, hold the motor command constant and vary actual displacement. |
| Hearing | A source produces a short nonverbal sound with left and right intensity. The world can generate these readings headlessly. | Hide the source, balance its side, then compare localization with intact and shuffled sound. |
| Tools and communication | Add held movement, pointing, interaction, keyboard focus and a writable surface through a game-defined control layout. Visible marks reach the existing camera as pixels. Characters may type arbitrary symbols. | First verify press, hold, release, edit and submit. Later give just one of two brains a private cue. Test the partner on fresh trials with intact, muted and shuffled messages while blocking other cues. |

Touch, movement feedback and hearing need explicit receiving pathways in the brain and extensions to the bridge. A collision can occur without punishment, and heat can be sensed before injury. Keep these facts distinct when defining sensors and outcomes. Introduce each pathway separately, verify that a stimulus reaches it, rerun the fork baseline and compare behavior with a matched copy in which the new input is silenced. The control layout already supports character entry. The world would supply a writing surface and a recipient, leaving vocabulary and meanings for the participants to develop. [Game control layouts](46-game-controls.md)

## Next

The first implementation milestone would be the fork, the local camera, a scripted controller and deterministic replay. The second would attach the existing bridge and measure the visual baseline. Wall contact is the first proposed brain-side addition. A second character can wait until there is a single-character baseline worth preserving. At that point, give each brain its own observations, collect both actions from the same tick and resolve them together under a documented collision rule.
