#!/usr/bin/env python3
"""
ASLR Visualizer 

Usage:
python3 aslr.py --program /path/to/program --runs 50 --duration 0.5 --out results

Notes:
- Regions produced: stack, heap, exec (main binary basename), and each major library (basename like libc.so.6).
- Kernel-provided [vdso]/[vvar] are recorded but not counted as mmap libraries for entropy.
"""
import argparse
import subprocess
import time
import os
import re
import math
import json
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Constants ---
PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')
MAPS_LINE_RE = re.compile(
    r"([0-9A-Fa-f]+)-([0-9A-Fa-f]+)\s+([rwxps-]{4})\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+:[0-9A-Fa-f]+)\s+(\d+)\s*(.*)"
)


# --- Utilities --------------------------------------------------------------

# Compiles the program if necessary and according to the specified pie enabled or disabled
def compile_if_source(program_path: str, pie_enabled: int) -> str:

    # Determine file type
    base, ext = os.path.splitext(program_path)
    if ext not in (".c", ".cpp"):
        return program_path   # Not a source file → use as-is

    # Choose compiler
    compiler = "gcc" if ext == ".c" else "g++"

    # Output executable (same base name)
    output_exe = base

    # PIE or non-PIE flags
    if pie_enabled == 0 or pie_enabled == -1:
        flags = ["-fno-pie", "-no-pie"]
        print(f"[+] Compiling {program_path} as NON-PIE → {output_exe}")
        
    else:
        flags = ["-fpie", "-pie"]
        print(f"[+] Compiling {program_path} as PIE → {output_exe}")

    # Compile command
    cmd = [compiler, program_path, "-o", output_exe] + flags

    # Run compiler
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("[-] Compilation failed:", e)
        exit(1)

    return output_exe


#Runs command to set aslr value mode: 0 = Off, 1 = Conservative/Partial, 2 = Full randomization
def set_aslr(mode: int):

    try:
        # Must run as root or with sudo
        subprocess.run(
            ["sudo", "sysctl", f"kernel.randomize_va_space={mode}"],
            check=True
        )
        print(f"[+] ASLR successfully set to {mode}")
    except subprocess.CalledProcessError:
        print("[-] Failed to change ASLR. Are you running with root permissions?")
    except FileNotFoundError:
        print("[-] sysctl not found. This command works only on Linux.")


def get_vm_range_from_maps():
    """
    Inspect /proc/self/maps to estimate the user-space VM range in use on this system.
    Returns (vm_start, vm_end). If something goes wrong, fall back to conservative defaults.
    """
    low = None
    high = None
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                addr = parts[0]
                if '-' not in addr:
                    continue
                start_hex, end_hex = addr.split('-', 1)
                try:
                    start = int(start_hex, 16)
                    end = int(end_hex, 16)
                except Exception:
                    continue
                if low is None or start < low:
                    low = start
                if high is None or end > high:
                    high = end
    except Exception:
        low = None
        high = None

    # Fallback if parsing failed
    if low is None or high is None or low >= high:
        # reasonable defaults for x86_64 user-space
        low = 0x0000000000400000
        high = 0x00007fffffffffff

    return low, high

def canonical_lib_name(pathname: str):
    """
    Normalize library/file pathnames into a canonical library key.
    Examples:
    /usr/lib/x86_64-linux-gnu/libc.so.6   -> libc.so.6
    /usr/lib/locale/C.utf8/LC_CTYPE       -> glibc-locale
    """
    if not pathname:
        return "anonymous"

    if pathname.startswith("[") and pathname.endswith("]"):
        # keep kernel virtual pages as-is (vdso, vvar, etc.)
        return pathname

    base = os.path.basename(pathname)

    # Merge locale fragments into a single key
    if "/locale/" in pathname or "/locales/" in pathname:
        return "glibc-locale"

    # If it's a .so (or contains .so.), return the name with extension
    if ".so" in base:
        # keep full name like libc.so.6
        idx = base.find(".so")
        # return the substring through .so plus possible suffix (e.g. .6)
        return base if base.endswith(".so") or base.count(".so") >= 1 else base

    # If it looks like an executed program (no .so), return basename as exec candidate
    return base

# --- Maps parsing -----------------------------------------------------------

def parse_maps(pid, exec_path=None):
    """
    Parse /proc/<pid>/maps and return OrderedDict of merged regions.

    Each key -> dict with:
    'start', 'end', 'size', 'perms', 'path', 'entries' (list of mapping segments)
    Keys are canonicalized as:
    - 'heap', 'stack'
    - '[vdso]', '[vvar]' (kept as-is)
    - exec basename if pathname == exec_path
    - canonical_lib_name(pathname) for file-backed mappings
    """
    path = f"/proc/{pid}/maps"
    regions = OrderedDict()
    try:
        with open(path, "r") as f:
            for line in f:
                m = MAPS_LINE_RE.match(line.strip())
                if not m:
                    continue
                start_hex, end_hex, perms, offset, dev, inode, pathname = m.groups()
                start = int(start_hex, 16)
                end = int(end_hex, 16)
                size = end - start
                pathname = pathname.strip()

                # Decide key
                if pathname == "[heap]":
                    key = "heap"
                elif pathname == "[stack]":
                    key = "stack"
                elif pathname in ("[vdso]", "[vvar]", "[vsyscall]"):
                    key = pathname
                elif pathname == "":
                    # anonymous mapping - group by "anon"
                    key = "anonymous"
                else:
                    # file-backed
                    # If exec_path provided and matches, label exec as basename
                    if exec_path is not None and os.path.abspath(pathname) == os.path.abspath(exec_path):
                        key = os.path.basename(pathname) or "exec"
                        # mark path as exec path explicitly
                    else:
                        key = canonical_lib_name(pathname)

                # Merge segments under same key
                if key not in regions:
                    regions[key] = {
                        "start": start,
                        "end": end,
                        "perms": perms,
                        "path": pathname,
                        "entries": []
                    }
                else:
                    regions[key]["start"] = min(regions[key]["start"], start)
                    regions[key]["end"] = max(regions[key]["end"], end)

                regions[key]["entries"].append({
                    "start": start,
                    "end": end,
                    "size": size,
                    "perms": perms,
                    "offset": offset,
                    "dev": dev,
                    "inode": inode,
                    "raw_path": pathname
                })

        # finalize sizes
        for key in regions:
            regions[key]["size"] = regions[key]["end"] - regions[key]["start"]

        return regions

    except FileNotFoundError:
        return None

# --- Runner ---------------------------------------------------------------

def run_program_and_capture(program, argv_list, duration):
    """
    Launch the program and sample /proc/<pid>/maps during runtime.
    Returns the merged regions dict (or None).
    """
    p = subprocess.Popen([program] + argv_list)
    pid = p.pid
    start_time = time.time()
    maps = None

    while True:
        current_maps = parse_maps(pid, exec_path=program)
        if current_maps:
            maps = current_maps
        if p.poll() is not None:
            break
        if time.time() - start_time > duration:
            break
        time.sleep(0.001)

    try:
        p.terminate()
        p.wait(timeout=1)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass

    return maps

# --- Stats helpers ---------------------------------------------------------

def bits_from_unique_count(addresses):
    unique = len(set(addresses))
    if unique <= 1:
        return 0.0
    return math.log2(unique)

def bits_from_range(addresses):
    if len(addresses) == 0:
        return 0.0
    mx = max(addresses)
    mn = min(addresses)
    slots = (mx - mn) // PAGE_SIZE + 1
    if slots <= 1:
        return 0.0
    return math.log2(slots)

def summarize_region(starts, sizes=None):
    """
    Given list of start addresses and optional sizes, produce summary dict.
    """
    arr = np.array(starts, dtype=np.int64)
    if arr.size == 0:
        return None
    rel = arr - arr[0]
    stats = {
        "count": int(arr.size),
        "unique": int(len(np.unique(arr))),
        "mean_offset": float(np.mean(rel)),
        "median_offset": float(np.median(rel)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "bits_unique": float(bits_from_unique_count(arr)),
        "bits_range": float(bits_from_range(arr)),
    }
    if sizes is not None and len(sizes) > 0:
        sarr = np.array(sizes, dtype=np.int64)
        stats.update({
            "size_count": int(sarr.size),
            "size_unique": int(len(np.unique(sarr))),
            "size_mean": float(np.mean(sarr)),
            "size_median": float(np.median(sarr)),
            "size_min": int(sarr.min()),
            "size_max": int(sarr.max()),
        })
    else:
        stats.update({
            "size_count": 0,
            "size_unique": 0,
            "size_mean": None,
            "size_median": None,
            "size_min": None,
            "size_max": None,
        })
    return stats

def analyze(all_runs):
    """
    Input: list of per-run region dicts (as returned by parse_maps)
    Output: data (starts per region), size_data (sizes per region), summaries
    """
    keys = set()
    for r in all_runs:
        if r:
            keys.update(r.keys())
    keys = sorted(keys)

    data = {k: [] for k in keys}
    size_data = {k: [] for k in keys}

    for r in all_runs:
        for k in keys:
            if r and k in r:
                data[k].append(r[k]["start"])
                size_data[k].append(r[k]["size"])
            else:
                data[k].append(None)
                size_data[k].append(None)

    summaries = {}
    for k in keys:
        starts = [s for s in data[k] if s is not None]
        sizes = [s for s in size_data[k] if s is not None]
        summaries[k] = summarize_region(starts, sizes)
    return data, size_data, summaries

def save_csv(data, outdir):
    df = pd.DataFrame({k: [v if v is not None else np.nan for v in vals] for k, vals in data.items()})
    df.to_csv(os.path.join(outdir, "addresses_per_run.csv"), index=False)

def make_histograms(summaries, outdir, threshold_low_bits=8.0):
    os.makedirs(outdir, exist_ok=True)
    flags = {}
    for k, s in summaries.items():
        if s is None:
            flags[k] = "missing"
            continue
        bits = min(s["bits_unique"], s["bits_range"])
        if bits == 0:
            flags[k] = "static"
        elif bits < threshold_low_bits:
            flags[k] = "low"
        else:
            flags[k] = "ok"
    with open(os.path.join(outdir, "flags.json"), "w") as f:
        json.dump(flags, f, indent=2)
    with open(os.path.join(outdir, "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    return flags

# --- Plots -----------------------------------------------------------------

def plot_runs_with_stats(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    for region, values in data.items():
        clean_vals = [v if v is not None else np.nan for v in values]
        arr = np.array(clean_vals, dtype=np.float64)
        if np.all(np.isnan(arr)):
            continue
        runs = list(range(1, len(clean_vals) + 1))
        mean_val = np.nanmean(arr)
        median_val = np.nanmedian(arr)
        plt.figure(figsize=(10, 4.5))
        plt.plot(runs, arr, "o-", label="Address per run", alpha=0.8)
        plt.axhline(mean_val, color="green", linestyle="--", label=f"Mean: {int(mean_val):#x}")
        plt.axhline(median_val, color="red", linestyle="-.", label=f"Median: {int(median_val):#x}")
        plt.xlabel("Run")
        plt.ylabel("Virtual Address")
        plt.title(f"ASLR addresses for region: {region}")
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(outdir, f"{region}_runs.png")
        plt.savefig(filename)
        plt.close()


def animate_address_changes(data, outdir, fps=4):
    """
    Creates per-region animated scatter plots showing address changes over runs.
    Produces: outdir/animations/<region>.mp4
    """
    anim_dir = os.path.join(outdir, "animations")
    os.makedirs(anim_dir, exist_ok=True)

    for region, values in data.items():
        clean_vals = [v if v is not None else np.nan for v in values]
        arr = np.array(clean_vals, dtype=np.float64)
        if np.all(np.isnan(arr)):
            continue

        runs = np.arange(1, len(arr) + 1)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.set_xlim(1, len(arr))
        ymin = np.nanmin(arr)
        ymax = np.nanmax(arr)
        if ymin == ymax:
            ymax += 1
        ax.set_ylim(ymin - (0.05 * abs(ymin)), ymax + (0.05 * abs(ymax)))

        ax.set_xlabel("Run")
        ax.set_ylabel("Address")
        ax.set_title(f"Address movement for region: {region}")

        pt, = ax.plot([], [], "o-", alpha=0.8)

        def init():
            pt.set_data([], [])
            return pt,

        def update(frame):
            # show up to current frame
            x = runs[: frame + 1]
            y = arr[: frame + 1]
            pt.set_data(x, y)
            return pt,

        frames = len(arr)
        interval = 1000 / fps  # ms per frame

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=frames,
            interval=interval,
            blit=True,
        )

        outpath = os.path.join(anim_dir, f"{region}.mp4")
        writer = FFMpegWriter(fps=fps, metadata={"artist": "ASLR Visualizer"})
        anim.save(outpath, writer=writer)

        plt.close(fig)


# --- Entropy & attack ------------------------------------------------------

def compute_total_entropy(summaries, program_name):
    """
    Compute E_s (stack), E_m (mmap), E_x (exec), E_h (heap)
    E_m is taken as max bits_range among libraries (.so or glibc-locale)
    E_x is taken from a key equal to program_name if present, else any non-.so candidate
    """
    E_s = summaries.get("stack", {}).get("bits_range", 0.0)
    E_h = summaries.get("heap", {}).get("bits_range", 0.0)

    # Exec candidate
    E_x = 0.0
    if program_name in summaries and summaries[program_name]:
        E_x = summaries[program_name].get("bits_range", 0.0)
    else:
        # any key that is not stack/heap/so/vdso/vvar and has stats can be considered exec
        for k, s in summaries.items():
            if k in ("stack", "heap", "[vdso]", "[vvar]", "anonymous"):
                continue
            if s is None:
                continue
            if ".so" not in k and not k.startswith("["):
                E_x = s.get("bits_range", 0.0)
                break

    # mmap: consider libraries keys containing .so or the canonical locale key
    so_bits = []
    for k, s in summaries.items():
        if s is None:
            continue
        if (".so" in k) or (k == "glibc-locale"):
            so_bits.append(s.get("bits_range", 0.0))
    E_m = max(so_bits) if so_bits else 0.0

    return E_s, E_m, E_x, E_h

def prob_isolated_guessing(alpha, N):
    if N <= 0:
        return 1.0 if alpha > 0 else 0.0
    return 1 - (1 - 2 ** (-N)) ** alpha

def prob_brute_force(alpha, N):
    if N <= 0:
        return 1.0 if alpha > 0 else 0.0
    max_attempts = 2 ** N
    if alpha > max_attempts:
        alpha = max_attempts
    return alpha / (2 ** N)

def compute_attack_and_guessing_stats(summaries, program_name, alpha=1000):
    E_s, E_m, E_x, E_h = compute_total_entropy(summaries, program_name)
    # Attack-reduced bits default to 0
    A_s = 0.0
    A_m = 0.0
    A_x = 0.0
    A_h = 0.0
    N = (E_s - A_s) + (E_m - A_m) + (E_x - A_x) + (E_h - A_h)
    g_alpha = prob_isolated_guessing(alpha, N)
    b_alpha = prob_brute_force(alpha, N)
    return {
        "E_s": E_s,
        "E_m": E_m,
        "E_x": E_x,
        "E_h": E_h,
        "A_s": A_s,
        "A_m": A_m,
        "A_x": A_x,
        "A_h": A_h,
        "N": N,
        "alpha": alpha,
        "prob_isolated_guessing": g_alpha,
        "prob_brute_force": b_alpha,
    }


# --- Additional stats ------------------------------------------------------

def compute_additional_stats(data):
    stats = {}
    region_keys = list(data.keys())

    for region in region_keys:
        arr = np.array(
            [v if v is not None else np.nan for v in data[region]],
            dtype=np.float64
        )
        valid = ~np.isnan(arr)
        valid_arr = arr[valid]

        # Detect zero variance or constant region
        constant = (valid_arr.size > 0 and np.allclose(valid_arr, valid_arr[0]))

        # Compute deltas only when possible
        deltas = np.diff(valid_arr) if valid_arr.size > 1 else np.array([])

        # Coefficient of variation:
        if not constant and np.nanmean(valid_arr) != 0:
            cv = float(np.nanstd(valid_arr) / np.nanmean(valid_arr))
        else:
            cv = None

        # Skewness: skip for constant data
        if not constant and valid_arr.size > 2:
            sk = float(skew(valid_arr, nan_policy="omit"))
        else:
            sk = None

        stats[region] = {
            "delta_mean": float(np.nanmean(deltas)) if deltas.size > 0 else None,
            "delta_std": float(np.nanstd(deltas)) if deltas.size > 0 else None,
            "delta_max": float(np.nanmax(deltas)) if deltas.size > 0 else None,
            "delta_min": float(np.nanmin(deltas)) if deltas.size > 0 else None,

            "coefficient_of_variation": cv,
            "skewness": sk,
        }

    # --------------------------
    # Correlation matrix
    # --------------------------
    corr_matrix = {}
    for r1 in region_keys:
        corr_matrix[r1] = {}
        arr1 = np.array([v if v is not None else np.nan for v in data[r1]], dtype=np.float64)
        valid1 = ~np.isnan(arr1)

        for r2 in region_keys:
            arr2 = np.array([v if v is not None else np.nan for v in data[r2]], dtype=np.float64)
            valid2 = ~np.isnan(arr2)
            mask = valid1 & valid2

            # Need >1 point AND non-zero variance in BOTH arrays
            if mask.sum() > 1:
                sub1 = arr1[mask]
                sub2 = arr2[mask]

                if np.std(sub1) > 0 and np.std(sub2) > 0:
                    corr = float(np.corrcoef(sub1, sub2)[0, 1])
                else:
                    corr = None
            else:
                corr = None

            corr_matrix[r1][r2] = corr

    return stats, corr_matrix




# --- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ASLR visualization across runs")
    p.add_argument("--program", required=True, help="Path to executable to run")
    p.add_argument("--args", nargs="*", default=[], help="Arguments to pass to program")
    p.add_argument("--runs", type=int, default=50, help="Number of runs to perform")
    p.add_argument("--duration", type=float, default=1, help="Seconds to wait before sampling /proc/<pid>/maps")
    p.add_argument("--out", default="aslr_results", help="Output directory")
    p.add_argument("--threshold-low-bits", type=float, default=20.0, help="Bits threshold below which randomness is 'low'")
    p.add_argument("--fps", type=int, default=4, help="Animation frames per second")
    p.add_argument("--aslr", type = int, default = 4, help = "OS ASLR Randomization Method. 0 = No randomization, 1 = Partial, 2= Full")
    p.add_argument("--pie", type = int, default = 1, help = "Program Independent Execution")
    
    
    args = p.parse_args()

    if args.aslr not in (0, 1, 2):
        raise ValueError("ASLR must be 0 (off), 1 (partial), or 2 (full)")

    print(f"[+] Requested ASLR mode: {args.aslr}")
    set_aslr(args.aslr)

    args.program = compile_if_source(args.program, args.pie)

    os.makedirs(args.out, exist_ok=True)

    # detect VM range 
    vm_start, vm_end = get_vm_range_from_maps()
    print(f"VM range: {hex(vm_start)} - {hex(vm_end)}")

    all_runs = []
    print(f"Running {args.runs} runs of {args.program} (duration {args.duration}s)")

    for i in range(args.runs):
        maps = run_program_and_capture(args.program, args.args, args.duration)
        if maps is None:
            print(f"Run {i+1}: process vanished before reading maps")
            all_runs.append({})
        else:
            all_runs.append(maps)
            print(f"Run {i+1}: captured {len(maps)} regions")

    # analyze
    data, size_data, summaries = analyze(all_runs)

    # program basename (exec name)
    program_name = os.path.basename(args.program)

    # save CSV and plots
    save_csv(data, args.out)
    plot_runs_with_stats(data, args.out)

    # histograms/flags & summaries
    flags = make_histograms(summaries, args.out, threshold_low_bits=args.threshold_low_bits)
    animate_address_changes(data, args.out, fps=args.fps)

    # report
    report = {
        "runs_requested": args.runs,
        "runs_recorded": len(all_runs),
        "page_size": PAGE_SIZE,
        "vm_start": vm_start,
        "vm_end": vm_end,
        "summaries": summaries,
        "flags": flags,
    }
    report["entropy"] = compute_attack_and_guessing_stats(summaries, program_name)

    additional_stats, correlations = compute_additional_stats(data)
    report["additional_stats"] = additional_stats
    report["correlations"] = correlations

    # write report
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # print summary
    ent = report["entropy"]
    print("\n=== ASLR Entropy Report ===")
    unique_regions = sorted(data.keys())
    print("\nUnique memory regions found:")
    for region in unique_regions:
        print(" -", region)
    print(f"Entropy bits (stack, mmap, exec, heap): {ent['E_s']}, {ent['E_m']}, {ent['E_x']}, {ent['E_h']}")
    print(f"Total entropy N: {ent['N']}")

    print("Done. Results are in:", args.out)
    print("Key files: addresses_per_run.csv, report.json, summaries.json, flags.json")

    set_aslr(2)

if __name__ == "__main__":
    main()
