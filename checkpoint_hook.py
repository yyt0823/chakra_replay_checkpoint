"""
Checkpoint metadata hook for PyTorch.

This module provides hooks to capture checkpoint metadata from torch.save() calls
and write it to a JSON file for use by the Chakra converter.
"""

import json
import os
import pickle
import threading
import torch

# Checkpoint metadata storage
_checkpoint_metadata = {}
_metadata_lock = threading.Lock()
_metadata_file_path = None

# Store original functions
_original_torch_save = torch.save
_original_serialization_save = torch.serialization._save

# Thread-local storage to pass checkpoint path from torch.save to _save
_thread_local = threading.local()


def torch_save_with_hook(obj, f, *args, **kwargs):
    """Hook for torch.save() to capture checkpoint path and extract metadata."""
    checkpoint_path = f if isinstance(f, str) else getattr(f, 'name', None)

    # Only collect logical metadata here (no file size yet)
    if isinstance(obj, dict) and checkpoint_path:
        metadata = {}
        if 'step' in obj:
            metadata['step'] = obj['step']
        if 'epoch' in obj:
            metadata['epoch'] = obj['epoch']
        if 'rank' in obj:
            metadata['rank'] = obj['rank']
        if 'batch_size' in obj:
            metadata['batch_size'] = obj['batch_size']
        metadata['checkpoint_path'] = checkpoint_path

        with _metadata_lock:
            entries = _checkpoint_metadata.setdefault(checkpoint_path, [])
            entries.append(metadata)

    # Just do the save; no size logic here
    return _original_torch_save(obj, f, *args, **kwargs)


def _compute_total_storage_bytes(path: str) -> int:
    """Return bytes used by this exact path (file or directory)."""
    # If path is a directory, sum everything under it
    if os.path.isdir(path):
        total_size = 0
        for root_dir, dirs, files in os.walk(path):
            for name in files:
                fp = os.path.join(root_dir, name)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
        return total_size

    # Otherwise, treat it as a single file
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def setup_checkpoint_hooks(rank, metadata_file_path=None):
    """
    Set up hooks and configure metadata output path for this rank.
    
    Args:
        rank (int): The rank/process ID for this process.
    """
    global _metadata_file_path
    if metadata_file_path is None:
        metadata_file_path = f"./result/checkpoint_metadata_{rank}.json"
    _metadata_file_path = metadata_file_path
    
    torch.save = torch_save_with_hook
    


def flush_checkpoint_metadata(rank):
    """
    Write checkpoint metadata to file for this rank.
    
    Args:
        rank (int): The rank/process ID for this process.
    """
    global _metadata_file_path, _checkpoint_metadata

    if _metadata_file_path and _checkpoint_metadata:
        with _metadata_lock:
            # AFTER all checkpoints are done, append storage info
            for checkpoint_path, entries in _checkpoint_metadata.items():
                try:
                    total_bytes = _compute_total_storage_bytes(checkpoint_path)
                except OSError:
                    total_bytes = None

                if total_bytes is not None:
                    for entry in entries:
                        entry['total_storage_bytes'] = total_bytes

            os.makedirs(os.path.dirname(_metadata_file_path) if os.path.dirname(_metadata_file_path) else '.', exist_ok=True)
            with open(_metadata_file_path, 'w') as f:
                json.dump(_checkpoint_metadata, f, indent=2)
            print(f"Checkpoint metadata written to {_metadata_file_path}")