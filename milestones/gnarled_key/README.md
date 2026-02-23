# Gnarled Key Milestone

**Achieved**: Epoch 9 (of the gate_slashed training run, Feb 23 2026)
**Model**: `model_epoch23.pt` — trained for 24 epochs total from gate_slashed save state
**Save State**: `epoch_9_gnarled_key.state` — PyBoy emulator state captured at the moment the Gnarled Key was obtained
**Mean Reward**: 16,907 (at time of milestone)
**Episodes**: 384 at 100% achievement rate

## What this represents

The agent learned to:
1. Start from the Maku Tree gate (sword already obtained)
2. Slash the gate with the sword
3. Navigate the Maku Tree grove
4. Pop the bubble around the Maku Tree
5. Advance through multi-box dialog to receive the Gnarled Key quest
6. Obtain the Gnarled Key item

This was the primary bottleneck in Oracle of Seasons progression. The Maku Tree dialog requires precise A-button timing across multiple text boxes — a behavior that RL agents typically struggle with.

## Training Configuration

- Save state: `save-states/advancing/epoch_23_gate_slashed.state`
- 24 environments, 8 workers
- 1,000,000 steps per epoch, 30,000 step episodes
- Dialog advance bonus: 25.0 (critical for teaching A-press during dialog)
- Gate slash bonus: 250.0, Maku dialog bonus: 500.0, Gnarled Key bonus: 500.0
