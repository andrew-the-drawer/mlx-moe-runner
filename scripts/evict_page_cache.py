#!/usr/bin/env python3
"""Evicts the file page cache by allocating and touching most of physical RAM.

macOS's F_NOCACHE reads bypass the cache for new data but do not evict pages
that are already cached (for example by a conversion that just wrote the expert
files), so benchmarks that claim to measure the SSD must run this first.
"""
import ctypes
import subprocess
import sys
import time

ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
fraction = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
target = int(ram * fraction)
chunk = 512 << 20
t0 = time.time()
blocks = []
touched = 0
try:
    while touched < target:
        b = ctypes.create_string_buffer(chunk)
        ctypes.memset(b, 1, chunk)
        blocks.append(b)
        touched += chunk
except MemoryError:
    pass
print(f"touched {touched / 1e9:.1f} GB in {time.time() - t0:.1f}s", file=sys.stderr)
blocks.clear()
