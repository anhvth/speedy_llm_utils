import os
import fcntl
import numpy as np
from loguru import logger
import tempfile
from typing import List

def _clear_port_use(ports):
    """
    Clear the usage counters for all ports.
    """
    for port in ports:
        file_counter = f"/tmp/port_use_counter_{port}.npy"
        if os.path.exists(file_counter):
            os.remove(file_counter)

def _atomic_save(array: np.ndarray, filename: str):
    """
    Write `array` to `filename` with an atomic rename to avoid partial writes.
    """
    # The temp file must be on the same filesystem as `filename` to ensure
    # that os.replace() is truly atomic.
    tmp_dir = os.path.dirname(filename) or "."
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp:
        np.save(tmp, array)
        temp_name = tmp.name

    # Atomically rename the temp file to the final name.
    # On POSIX systems, os.replace is an atomic operation.
    os.replace(temp_name, filename)

def _update_port_use(port: int, increment: int):
    """
    Update the usage counter for a given port, safely with an exclusive lock
    and atomic writes to avoid file corruption.
    """
    file_counter = f"/tmp/port_use_counter_{port}.npy"
    file_counter_lock = f"/tmp/port_use_counter_{port}.lock"

    with open(file_counter_lock, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # If file exists, load it. Otherwise assume zero usage.
            if os.path.exists(file_counter):
                try:
                    counter = np.load(file_counter)
                except Exception as e:
                    # If we fail to load (e.g. file corrupted), start from zero
                    logger.warning(f"Corrupted usage file {file_counter}: {e}")
                    counter = np.array([0])
            else:
                counter = np.array([0])

            # Increment usage and atomically overwrite the old file
            counter[0] += increment
            _atomic_save(counter, file_counter)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def _pick_least_used_port(ports: List[int]) -> int:
    """
    Pick the least-used port among the provided list, safely under a global lock
    so that no two processes pick a port at the same time.
    """
    global_lock_file = "/tmp/ports.lock"

    with open(global_lock_file, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            port_use = {}
            # Read usage for each port
            for port in ports:
                file_counter = f"/tmp/port_use_counter_{port}.npy"
                if os.path.exists(file_counter):
                    try:
                        counter = np.load(file_counter)
                    except Exception as e:
                        # If the file is corrupted, reset usage to 0
                        logger.warning(f"Corrupted usage file {file_counter}: {e}")
                        counter = np.array([0])
                else:
                    counter = np.array([0])
                port_use[port] = counter[0]

            logger.debug(f"Port use: {port_use}")

            # Pick the least-used port
            lsp = min(port_use, key=port_use.get)

            # Increment usage of that port
            _update_port_use(lsp, 1)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return lsp
