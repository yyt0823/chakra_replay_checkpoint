# Chakra Replay — Checkpoint Extension

Extends [Chakra](https://github.com/facebookresearch/chakra) and [ET-Replay](https://github.com/facebookresearch/param) with **checkpoint-aware execution trace recording and simulation** — enabling accurate replay of distributed ML training workloads that include model checkpointing.

> **Skills:** PyTorch · Distributed training · Execution trace simulation · Chakra · ET-Replay · Python hooks · JSON metadata · CLI tooling

---

## Overview

Chakra traces are used to simulate and replay the compute/communication patterns of large-scale training runs. However, standard traces don't capture checkpoint operations (`torch.save`), which are a significant and irregular I/O cost in real training.

This project adds:
- A **checkpoint hook** that intercepts `torch.save()` calls and captures metadata (step, epoch, rank, file size, timing) into a JSON sidecar file
- A **new `CHECKPOINT_NODE` type** in the Chakra execution trace schema
- A **modified ET-Replay engine** that can simulate checkpoint I/O during replay
- A **unified CLI** (`full_run_cli`) to run the full pipeline end-to-end

---

## Demo

Full walkthrough: [YouTube](https://youtu.be/IR1J9kq53H0)

![demo](demo.gif)

---

## How to Use

### 1. Import the Checkpoint Hook

In your training or profiling script:

```python
from checkpoint_hook import setup_checkpoint_hooks, flush_checkpoint_metadata
```

### 2. Set Up Hooks Before Training

Initialize once per rank before training starts:

```python
setup_checkpoint_hooks(rank)
```

After training, flush the recorded metadata to disk:

```python
flush_checkpoint_metadata(rank)
```

### 3. Replace Chakra and ET-Replay

Use the patched versions included in this repo:

- `chakra/` — extended Chakra with `CHECKPOINT_NODE` support
- `et_replay/` — modified replay engine that handles checkpoint simulation

### 4. Run the Full Pipeline

```bash
full_run_cli
```

---

## Project Structure

| File / Folder | Description |
|---------------|-------------|
| `checkpoint_hook.py` | PyTorch hook that intercepts `torch.save()` and logs checkpoint metadata to JSON |
| `chakra/` | Patched Chakra with `CHECKPOINT_NODE` added to the execution trace schema |
| `et_replay/` | Modified ET-Replay engine supporting checkpoint node simulation |
| `full_run_cml` | End-to-end CLI for running the full record → convert → replay pipeline |
| `test.py` | Test script for validating checkpoint hook and trace output |
| `Checkpoint Replay Support for Chakra and ET-Replay.pdf` | Full technical writeup |
