# Adaptive-FPS Sensing — SMDP Formulation (CarRacing)

## Research question
Can a learned policy dynamically select observation frequency (FPS) to
minimize perceptual sampling cost while still completing the driving task?
A frozen navigation controller (NavModel) drives the car; a separate policy
learns to choose how often NavModel gets a fresh observation to drive with.

## How the SMDP mechanism works
- In the fixed-FPS baseline, one `env.step()` = one frame, and NavModel gets
  a fresh observation every frame.
- In the adaptive-FPS task, the FPS policy chooses a sampling rate. That
  choice defines a **decision window**: a run of frames over which the FPS
  policy doesn't act again — NavModel drives through the window using
  whatever observation policy the chosen FPS implies, and a new FPS decision
  is only made once the window ends.
- Because windows can be different lengths depending on the FPS chosen, one
  "step" from the FPS policy's perspective is not a fixed unit of time —
  this is what makes it a semi-Markov decision process (SMDP) rather than a
  standard MDP.
- This has two direct consequences for training:
  1. **Reward accumulation within a window** — the frames inside one window
     produce a sequence of rewards that need to be combined (discounted)
     into a single value credited to the FPS decision that caused that
     window.
  2. **Discounting between windows** — since windows vary in length, the
     discount applied between one FPS decision and the next must scale with
     how long the window was (`gamma ** duration`), not a fixed `gamma` per
     decision.
- The FPS policy's reward also needs a cost term that scales with sampling
  rate, so it's pushed toward fewer observations rather than just toward
  best driving performance.

## References to study before implementing
1. **LunarLander adaptive-FPS implementation** lunarlander_env_ref.py and lunarlander_train_ref.py. 
   Working recurrent PPO implementation of variable sensing
   frequency with a frozen controller. Use this for the general pattern of
   how the FPS action, window structure, and reward cost term were wired
   together.
2. **CarRacing Highest FPS Experiment** — `wrappers/highest_fps_cautious_step_loop.py`.
   Prior attempt at this exact task on CarRacing. Read it
   alongside the references: envs/car_racing_var_fps.py and the wrappers: wrappers/pre_processing.py, utils/cautious_variables.py. Where it agrees with the LunarLander
   pattern and the SMDP mechanism above, treat it as a useful starting point;

## Task
Adapt the CarRacing Highest FPS baseline into the adaptive-FPS SMDP version,
using it as the pattern for how the sensing
policy and cost term are structured.

Concretely:
1. Make a new adaptive wrapper adding the sampling-cost term to the reward and what else do you need is important to obtain the target policy behavior (adaptive fps).
2. Make sure the sampling cost added is enough to guarante the adaptiveness we are looking for. If memory is needed, add it. Otherwise, make it enough to obtain the target policy behavior.
3. Create a training file in a new folder under experiments.
4. Keep the same running/saving/logging mechanisms as the folders under experiments/
5. Verify NavModel is driven correctly during a window (using whatever
   observation-holding behavior the chosen FPS implies)
6. Before running full training, instrument a short rollout with asserts
   confirming that the action, reward span, and duration stored per buffer
   index all correspond to the same window.

## Setup
- Conda env: `racing` (PyTorch, CleanRL-based PPO)
- Cluster: `ece-focus-xg02`, 8x RTX A6000
- Config management: `tyro` / YAML
- Tracking: WandB + TensorBoard