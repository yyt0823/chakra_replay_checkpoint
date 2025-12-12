# chakra_replay_checkpoint
extend chakra with checkpointing recording and simulation

how to use


1. import the checkpoint_hook
from checkpoint_hook import setup_checkpoint_hooks, flush_checkpoint_metadata

2. set up for the hook:  \n use this to set up hooks: setup_checkpoint_hooks(rank) \n and this to flush : flush_checkpoint_metadata(rank)

3. replace the chakra and et_replay and use the new command line in the full_run_cml file
