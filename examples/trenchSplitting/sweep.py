#!/usr/bin/env python3
"""
sweep.py — systematic parameter sweep for depth-adaptive ray splitting.

Iterates over (AR, sticking, numSplits, splitFactor), calls the trenchSplitting
binary for each combination, and records the worst depth-bin surface-hit count
per seed per second (baseline and split) to a CSV.  Results are written after
every run so the file is usable if the sweep is interrupted.

Usage:
    python3 sweep.py [options]

Options:
    --binary PATH     Path to trenchSplitting executable
                      (default: ../../build/examples/trenchSplitting/trenchSplitting)
    --out PATH        Output CSV  (default: sweep_results.csv)
    --dim 2|3         Geometry: 2=periodic trench, 3=cylindrical hole  (default: 2)
    --workers N       Parallel runs  (default: auto = nproc // omp_threads)
    --threads N       OMP threads per run  (default: 2)
    --gpu             Run the GPU path. Defaults to one worker to avoid
                      oversubscribing one CUDA device.
    --rays-per-point N
    --seeds N         Independent seeds for N_eff estimation.

Parallel workers × threads should not exceed the physical core count.

Example — 2D sweep with 11 workers × 2 threads on a 22-core machine:
    python3 sweep.py --dim 2 --workers 11 --threads 2

Then run the 3D sweep into a separate file:
    python3 sweep.py --dim 3 --workers 11 --threads 2 --out sweep_results_3d.csv
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
ASPECT_RATIOS = [3, 5, 7, 10, 15, 20, 30, 50, 70, 100]
STICKINGS     = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
NUM_SPLITS    = [2, 3, 5, 7, 10]
SPLIT_FACTORS = [2, 3, 4, 5]

# Fixed geometry / simulation parameters.  The defaults are intentionally close
# to the validated CPU/GPU equivalence case; override them from the notebook or
# CLI for faster exploratory sweeps.
FEATURE_SIZE   = 0.10    # µm — half-width (2D) or radius (3D)
HALF_CELL      = 0.30    # µm — must exceed FEATURE_SIZE; sets XY periodic boundary
GRID_DELTA     = 0.02    # µm
CONE_ANGLE_DEG = 90.0    # degrees — isotropic cosine source
SOURCE_POWER   = 1.0
RAYS_PER_PT    = 30      # primary rays per surface primitive
N_SEEDS        = 50      # independent seeds for stable N_eff estimation
PROBE_RAYS     = 20      # rays/pt in the SplittingStrategy probe pass
PROBE_SEEDS    = 5
KILL_FRAC      = 0.0001  # RR weight floor (fraction of initial weight)
MAX_BOUNDARY   = 100000


# ---------------------------------------------------------------------------
# Config file generation
# ---------------------------------------------------------------------------
def make_config(dim: int, ar: int, sticking: float,
                num_splits: int, split_factor: int, threads: int,
                use_gpu: bool = False, rays_per_point: int = RAYS_PER_PT,
                n_seeds: int = N_SEEDS, probe_rays: int = PROBE_RAYS,
                probe_seeds: int = PROBE_SEEDS,
                kill_frac: float = KILL_FRAC) -> str:
    depth = FEATURE_SIZE * ar
    entries = {
        "dimension":         dim,
        "useGPU":            1 if use_gpu else 0,
        "depth":             f"{depth:.4f}",
        "halfCell":          f"{HALF_CELL:.4f}",
        "gridDelta":         f"{GRID_DELTA:.4f}",
        "sticking":          f"{sticking:.4f}",
        "coneAngle":         f"{CONE_ANGLE_DEG:.1f}",
        "sourcePower":       f"{SOURCE_POWER:.1f}",
        "raysPerPoint":      str(rays_per_point),
        "nSeeds":            str(n_seeds),
        "numSplits":         str(num_splits),
        "splitFactor":       str(split_factor),
        "splitKillFraction": f"{kill_frac}",
        "probeRays":         str(probe_rays),
        "probeSeeds":        str(probe_seeds),
        "numThreads":        str(threads),
        "maxBoundaryHits":   str(MAX_BOUNDARY),
    }
    if dim == 2:
        entries["width"]  = f"{FEATURE_SIZE:.4f}"
    else:
        entries["radius"] = f"{FEATURE_SIZE:.4f}"
    return "\n".join(f"{k} = {v}" for k, v in entries.items())


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------
def parse_output(text: str) -> dict | None:
    """Extract metrics from one binary run.  Returns None if output is malformed."""
    r = {}

    # Legacy bottom N_eff efficiency lines.  Kept for backwards compatibility
    # with older trenchSplitting binaries; new output is scored by raw hit count.
    #   baseline:  1902.5  N_eff/s  (N_eff_bottom = 23)
    #   split:     4367.9  N_eff/s  (N_eff_bottom = 31)
    m = re.search(
        r"baseline:\s+([\d.]+)\s+N_eff/s\s+\(N_eff_bottom\s*=\s*(\d+)\)", text
    )
    if m:
        r["eff_base"]      = float(m.group(1))
        r["neff_bot_base"] = int(m.group(2))
    else:
        r["eff_base"]      = float("nan")
        r["neff_bot_base"] = 0

    m = re.search(
        r"split:\s+([\d.]+)\s+N_eff/s\s+\(N_eff_bottom\s*=\s*(\d+)\)", text
    )
    if m:
        r["eff_split"]      = float(m.group(1))
        r["neff_bot_split"] = int(m.group(2))
    else:
        r["eff_split"]      = float("nan")
        r["neff_bot_split"] = 0

    # Wall times: two "total time = X.XX s" lines (baseline first, split second)
    times = re.findall(r"total time\s*=\s*([\d.]+)\s*s", text)
    r["time_base"]  = float(times[0]) if len(times) > 0 else float("nan")
    r["time_split"] = float(times[1]) if len(times) > 1 else float("nan")

    # Per-depth N_eff rows.  These are the stochastic-quality metrics: they
    # estimate the effective number of independent hits contributing to flux.
    # The quality selector is the least-sampled bin, independently for baseline
    # and split.  This is the stochastic improvement target.
    neff_section = ""
    m_neff_section = re.search(
        r"Effective ray count per wall primitive(?P<section>.*?)(?:Surface hit count per depth bin|Mean normalized flux per wall primitive|$)",
        text,
        flags=re.DOTALL,
    )
    if m_neff_section:
        neff_section = m_neff_section.group("section")
    neff_rows = re.findall(
        r"^\s*([\d.]+)\s*-\s*([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)x\s*$",
        neff_section,
        flags=re.MULTILINE,
    )
    neff_records = []
    for y0, y1, nb, ns, ratio in neff_rows:
        neff_records.append(
            (round(float(y0), 6), round(float(y1), 6),
             int(nb), int(ns), float(ratio))
        )

    # Per-depth raw surface-hit rows.  This is the primary stochastic-quality
    # metric: literal unweighted collision counts in each bin, reported per
    # seed.  The quality selector is the least-hit bin, independently for
    # baseline and split.
    hit_section = ""
    m_hit_section = re.search(
        r"Surface hit count per depth bin(?P<section>.*?)(?:Mean normalized flux per wall primitive|$)",
        text,
        flags=re.DOTALL,
    )
    if m_hit_section:
        hit_section = m_hit_section.group("section")
    hit_rows = re.findall(
        r"^\s*([\d.]+)\s*-\s*([\d.]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)x\s*$",
        hit_section,
        flags=re.MULTILINE,
    )
    hit_records = []
    for y0, y1, hb, hs, ratio in hit_rows:
        hit_records.append(
            (round(float(y0), 6), round(float(y1), 6),
             float(hb), float(hs), float(ratio))
        )

    # Detected axis info from SplittingStrategy log line:
    #   [SplittingStrategy] depth axis = Y  range = 1.00  splitInterval = 0.20 ...
    m = re.search(
        r"\[(?:gpu::)?SplittingStrategy\]\s+depth axis\s*=\s*(\w+)"
        r"\s+range\s*=\s*([\d.]+)"
        r"\s+splitInterval\s*=\s*([\d.]+)",
        text,
    )
    if m:
        r["detected_axis"]  = m.group(1)
        r["geometry_range"] = float(m.group(2))
        r["split_interval"] = float(m.group(3))
    else:
        r["detected_axis"]  = ""
        r["geometry_range"] = float("nan")
        r["split_interval"] = float("nan")

    m = re.search(
        r"(?:GPU\s+)?Splitting\s+\(axis\s*=\s*(\d+),\s*interval\s*=\s*([\d.eE+-]+)",
        text,
    )
    if m:
        r["split_axis"] = int(m.group(1))
        r["split_interval"] = float(m.group(2))
    else:
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        r["split_axis"] = axis_map.get(r["detected_axis"], -1)

    # Mean N_eff across all depth bins (σ/mean uniformity metric)
    # "σ/mean:  baseline = 1.332   split = 1.324"
    m = re.search(r"[σs]/mean:\s+baseline\s*=\s*([\d.]+)\s+split\s*=\s*([\d.]+)", text)
    if m:
        r["cv_base"]  = float(m.group(1))   # coefficient of variation (lower = more uniform)
        r["cv_split"] = float(m.group(2))
    else:
        r["cv_base"]  = float("nan")
        r["cv_split"] = float("nan")

    # Warning flags
    r["base_starved"] = (
        "baseline cannot reach the bottom" in text or
        "baseline has an empty worst bin" in text
    )
    r["both_starved"] = (
        "neither strategy reaches the bottom" in text or
        "neither strategy records hits in its worst bin" in text
    )

    # Mean flux table.  Flux is retained only as an optional bias sanity check;
    # it is not used to select or score stochastic quality.
    flux_section = ""
    m_flux_section = re.search(
        r"Mean normalized flux per wall primitive(?P<section>.*?)(?:Bottom N_eff efficiency|Worst-bin hit efficiency|$)",
        text,
        flags=re.DOTALL,
    )
    if m_flux_section:
        flux_section = m_flux_section.group("section")
    flux_rows = re.findall(
        r"^\s*([\d.eE+-]+)\s*-\s*([\d.eE+-]+)\s+"
        r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)x\s*$",
        flux_section,
        flags=re.MULTILINE,
    )
    if flux_rows:
        flux_records = [
            (round(float(y0), 6), round(float(y1), 6),
             float(fb), float(fs), float(ratio))
            for y0, y1, fb, fs, ratio in flux_rows
        ]
        _, _, fb, fs, ratio = flux_records[-1]  # retained for backwards compatibility
        _, _, min_fb, min_fs, min_ratio = min(flux_records, key=lambda x: x[2])
        r["flux_bot_base"] = fb
        r["flux_bot_split"] = fs
        r["flux_bot_ratio"] = ratio
        r["flux_min_base"] = min_fb
        r["flux_min_split"] = min_fs
        r["flux_min_ratio"] = min_ratio
    else:
        r["flux_bot_base"] = float("nan")
        r["flux_bot_split"] = float("nan")
        r["flux_bot_ratio"] = float("nan")
        r["flux_min_base"] = float("nan")
        r["flux_min_split"] = float("nan")
        r["flux_min_ratio"] = float("nan")

    if hit_records:
        base_min = min(hit_records, key=lambda x: x[2])
        split_min = min(hit_records, key=lambda x: x[3])
        r["hit_min_base"] = base_min[2]
        r["hit_min_split"] = split_min[3]
        r["hit_min_base_depth0"] = base_min[0]
        r["hit_min_base_depth1"] = base_min[1]
        r["hit_min_split_depth0"] = split_min[0]
        r["hit_min_split_depth1"] = split_min[1]
    else:
        r["hit_min_base"] = 0
        r["hit_min_split"] = 0
        r["hit_min_base_depth0"] = float("nan")
        r["hit_min_base_depth1"] = float("nan")
        r["hit_min_split_depth0"] = float("nan")
        r["hit_min_split_depth1"] = float("nan")

    r["hit_min_ratio"] = (
        r["hit_min_split"] / r["hit_min_base"]
        if r["hit_min_base"] > 0 else float("inf")
    )
    hit_eff_base = hit_eff_split = None
    m = re.search(
        r"baseline:\s+([\d.eE+-]+)\s+hits/(?:seed/)?s\s+\(min_hits(?:_per_seed)?\s*=\s*([\d.eE+-]+)",
        text,
    )
    if m:
        hit_eff_base = float(m.group(1))
        r["hit_min_base"] = float(m.group(2))
    m = re.search(
        r"split:\s+([\d.eE+-]+)\s+hits/(?:seed/)?s\s+\(min_hits(?:_per_seed)?\s*=\s*([\d.eE+-]+)",
        text,
    )
    if m:
        hit_eff_split = float(m.group(1))
        r["hit_min_split"] = float(m.group(2))

    if neff_records:
        base_min = min(neff_records, key=lambda x: x[2])
        split_min = min(neff_records, key=lambda x: x[3])
        r["neff_min_base"] = base_min[2]
        r["neff_min_split"] = split_min[3]
        r["neff_min_base_depth0"] = base_min[0]
        r["neff_min_base_depth1"] = base_min[1]
        r["neff_min_split_depth0"] = split_min[0]
        r["neff_min_split_depth1"] = split_min[1]
    else:
        r["neff_min_base"] = 0
        r["neff_min_split"] = 0
        r["neff_min_base_depth0"] = float("nan")
        r["neff_min_base_depth1"] = float("nan")
        r["neff_min_split_depth0"] = float("nan")
        r["neff_min_split_depth1"] = float("nan")

    r["neff_min_ratio"] = (
        r["neff_min_split"] / r["neff_min_base"]
        if r["neff_min_base"] > 0 else float("inf")
    )

    use_hits = bool(hit_records)
    quality_base = r["hit_min_base"] if use_hits else r["neff_min_base"]
    quality_split = r["hit_min_split"] if use_hits else r["neff_min_split"]
    quality_ratio = r["hit_min_ratio"] if use_hits else r["neff_min_ratio"]

    r["quality_metric"] = "hits_per_seed" if use_hits else "neff"
    r["neff_quality_base"] = quality_base
    r["neff_quality_split"] = quality_split
    r["neff_quality_ratio"] = quality_ratio
    r["eff_quality_base"] = quality_base / r["time_base"] if r["time_base"] > 0 else float("nan")
    r["eff_quality_split"] = quality_split / r["time_split"] if r["time_split"] > 0 else float("nan")
    if use_hits and hit_eff_base is not None:
        r["eff_quality_base"] = hit_eff_base
    if use_hits and hit_eff_split is not None:
        r["eff_quality_split"] = hit_eff_split
    r["speedup"] = (
        r["eff_quality_split"] / r["eff_quality_base"]
        if r["eff_quality_base"] > 0 else float("inf")
    )

    return r if hit_records or neff_records else None


# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------
_csv_lock = threading.Lock()


def run_one(binary: Path, out_path: Path, writer: csv.DictWriter,
            fieldnames: list[str], dim: int, ar: int, sticking: float,
            num_splits: int, split_factor: int, threads: int,
            idx: int, total: int, t0_all: float,
            use_gpu: bool = False, rays_per_point: int = RAYS_PER_PT,
            n_seeds: int = N_SEEDS, probe_rays: int = PROBE_RAYS,
            probe_seeds: int = PROBE_SEEDS, kill_frac: float = KILL_FRAC,
            timeout_s: int = 600) -> dict | None:
    cfg_text = make_config(dim, ar, sticking, num_splits, split_factor, threads,
                           use_gpu=use_gpu, rays_per_point=rays_per_point,
                           n_seeds=n_seeds, probe_rays=probe_rays,
                           probe_seeds=probe_seeds, kill_frac=kill_frac)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(cfg_text)
        cfg_path = Path(f.name)

    t0 = time.time()
    label = f"ar={ar:2d} s={sticking:.2f} ns={num_splits} sf={split_factor}"
    try:
        proc = subprocess.run(
            [str(binary), str(cfg_path)],
            capture_output=True, text=True, timeout=timeout_s,
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        with _csv_lock:
            print(f"  [{idx:4d}/{total}]  {label}  TIMEOUT ({elapsed:.0f}s)", flush=True)
        cfg_path.unlink(missing_ok=True)
        return None
    finally:
        cfg_path.unlink(missing_ok=True)

    parsed = parse_output(proc.stdout)
    if parsed is None:
        with _csv_lock:
            print(f"  [{idx:4d}/{total}]  {label}  PARSE ERROR", flush=True)
            if proc.stderr:
                print("   stderr:", proc.stderr[:300], flush=True)
        return None

    row = {
        "backend": "gpu" if use_gpu else "cpu",
        "dim": dim, "ar": ar, "sticking": sticking,
        "num_splits": num_splits, "split_factor": split_factor,
        "rays_per_point": rays_per_point, "n_seeds": n_seeds,
        "probe_rays": probe_rays, "probe_seeds": probe_seeds,
        "kill_frac": kill_frac,
        **parsed,
    }

    done = idx
    elapsed_all = time.time() - t0_all
    eta_s = elapsed_all / done * (total - done) if done > 0 else 0
    flag = ("  [BOTH STARVED]" if parsed["both_starved"] else
            "  [BASE STARVED]" if parsed["base_starved"] else "")

    with _csv_lock:
        writer.writerow(row)
        # flush happens via the open file handle in main
        print(
            f"  [{idx:4d}/{total}]  {label}  "
            f"speedup={parsed['speedup']:5.2f}x  "
            f"(run {elapsed:.1f}s  ETA {eta_s/60:.1f} min){flag}",
            flush=True,
        )

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--binary",
        default=str(
            Path(__file__).parent / "../../build/examples/trenchSplitting/trenchSplitting"
        ),
        help="Path to trenchSplitting executable",
    )
    ap.add_argument("--out",     default="sweep_results.csv")
    ap.add_argument("--dim",     type=int, default=2, choices=[2, 3])
    ap.add_argument("--threads", type=int, default=2,
                    help="OMP threads per run")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel runs (default: nproc // threads)")
    ap.add_argument("--gpu", action="store_true",
                    help="Use GPU path (requires CUDA build; supports dim=2 and dim=3)")
    ap.add_argument("--rays-per-point", type=int, default=RAYS_PER_PT)
    ap.add_argument("--seeds", type=int, default=N_SEEDS,
                    help="Independent seeds used to estimate N_eff")
    ap.add_argument("--probe-rays", type=int, default=PROBE_RAYS)
    ap.add_argument("--probe-seeds", type=int, default=PROBE_SEEDS)
    ap.add_argument("--kill-frac", type=float, default=KILL_FRAC)
    ap.add_argument("--timeout", type=int, default=600,
                    help="Timeout per binary run in seconds")
    args = ap.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        sys.exit(f"Binary not found: {binary}\n"
                 f"Build first:  cmake --build <build-dir> --target trenchSplitting")

    use_gpu = args.gpu
    n_workers = args.workers or (1 if use_gpu else max(1, os.cpu_count() // args.threads))

    combos = list(itertools.product(ASPECT_RATIOS, STICKINGS, NUM_SPLITS, SPLIT_FACTORS))
    total  = len(combos)
    print(f"Sweep: {total} combinations  dim={args.dim}  gpu={use_gpu}  "
          f"workers={n_workers}  omp_threads={args.threads}")
    print(f"Output: {args.out}\n")

    fieldnames = [
        "backend", "dim", "ar", "sticking", "num_splits", "split_factor",
        "rays_per_point", "n_seeds", "probe_rays", "probe_seeds", "kill_frac",
        "quality_metric",
        "speedup",
        "hit_min_base", "hit_min_split", "hit_min_ratio",
        "hit_min_base_depth0", "hit_min_base_depth1",
        "hit_min_split_depth0", "hit_min_split_depth1",
        "eff_quality_base", "eff_quality_split",
        "time_base", "time_split",
        "flux_min_ratio",
        "split_axis", "split_interval",
        "base_starved", "both_starved",
    ]

    out_path = Path(args.out)
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    t0_all = time.time()
    with open(out_path, "a", newline="", buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    run_one,
                    binary, out_path, writer, fieldnames,
                    args.dim, ar, sticking, ns, sf, args.threads,
                    idx, total, t0_all, use_gpu, args.rays_per_point,
                    args.seeds, args.probe_rays, args.probe_seeds,
                    args.kill_frac, args.timeout,
                ): (ar, sticking, ns, sf)
                for idx, (ar, sticking, ns, sf) in enumerate(combos, 1)
            }
            for _ in as_completed(futures):
                pass   # progress is printed inside run_one

    total_time = time.time() - t0_all
    print(f"\nDone in {total_time/60:.1f} min.  Results: {args.out}")


if __name__ == "__main__":
    main()
