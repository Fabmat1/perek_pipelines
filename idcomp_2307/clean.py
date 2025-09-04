#!/usr/bin/env python3
import glob
import os

def trim_file(fname: str):
    """Keep only the lines from the last occurrence of 'aplow' onward."""
    with open(fname, "r") as f:
        lines = f.readlines()

    last_idx = None
    for i, line in enumerate(lines):
        if "aplow" in line:
            last_idx = i

    if last_idx is not None:
        new_lines = lines[last_idx:]
        with open(fname, "w") as f:
            f.writelines(new_lines)
        print(f"Processed: {fname}")
    else:
        print(f"Skipped (no 'aplow' found): {fname}")

def main():
    for fname in sorted(glob.glob("idiazcomp.*")):
        if os.path.isfile(fname):
            trim_file(fname)

if __name__ == "__main__":
    main()
