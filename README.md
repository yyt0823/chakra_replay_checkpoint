# chakra_replay_checkpoint
extend chakra with checkpointing recording and simulation

How to Use
1. Import the Checkpoint Hook

In your training or profiling script, import the checkpoint hook utilities:

from checkpoint_hook import setup_checkpoint_hooks, flush_checkpoint_metadata

2. Set Up and Flush Checkpoint Hooks

Initialize the checkpoint hooks once per rank before training starts:

setup_checkpoint_hooks(rank)


After training (or at the end of profiling), flush the recorded checkpoint metadata:

flush_checkpoint_metadata(rank)

3. Replace Chakra and ET-Replay

Use the patched versions of Chakra and ET-Replay that include checkpoint support.

Replace your existing Chakra installation with the extended version

Replace ET-Replay with the modified replay engine that supports CHECKPOINT_NODE

4. Run Using the New CLI

Use the updated command-line interface defined in:

full_run_cli



## Demo

![demo1](demo1.gif)
![demo2](demo2.gif)
![demo3](demo3.gif)
![demo4](demo4.gif)
![demo5](demo5.gif)
![demo6](demo6.gif)
![demo7](demo7.gif)
![demo8](demo8.gif)
![demo9](demo9.gif)
![demo10](demo10.gif)
![demo11](demo11.gif)
![demo12](demo12.gif)
![demo13](demo13.gif)
![demo14](demo14.gif)
![demo15](demo15.gif)
