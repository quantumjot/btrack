#!/usr/bin/env python
"""Benchmark script for hypothesis generation performance analysis."""
# ruff: noqa: T201, E501, RUF001

import btrack
from btrack import datasets

import time
from collections import Counter

import numpy as np


def benchmark_hypothesis_generation(n_objects=100, n_frames=50):
    """Benchmark the hypothesis generation and optimization process."""

    print(f"Creating test data: {n_objects} objects across {n_frames} frames...")

    # Create synthetic tracking data
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(0, 1000, n_objects * n_frames)
    y = rng.uniform(0, 1000, n_objects * n_frames)
    z = np.zeros_like(x)
    t = np.repeat(np.arange(n_frames), n_objects)

    objects_dict = {"x": x, "y": y, "z": z, "t": t}
    objects = btrack.io.objects_from_dict(objects_dict)

    # Set up tracker
    print("Setting up tracker...")
    tracker = btrack.BayesianTracker()
    config = btrack.config.load_config(datasets.cell_config())
    config.motion_model.prob_not_assign = 0.001
    tracker.configure(config)
    tracker.append(objects)
    tracker.volume = ((0, 1200), (0, 1200), (-1e5, 1e5))

    # Run tracking
    print("Running tracking...")
    t_track_start = time.perf_counter()
    tracker.track(step_size=100)
    t_track = time.perf_counter() - t_track_start
    print(f"  Tracking completed in {t_track:.3f}s")
    print(f"  Found {tracker.n_tracks} tracks")

    # Generate hypotheses (this is what we want to profile)
    print("\nGenerating hypotheses...")
    t_hyp_start = time.perf_counter()
    hypotheses = tracker.hypotheses()
    t_hyp = time.perf_counter() - t_hyp_start
    print(f"  Hypothesis generation: {t_hyp:.3f}s")
    print(f"  Generated {len(hypotheses)} hypotheses")

    # Count hypothesis types
    h_types = Counter([h.hypothesis_type for h in hypotheses])
    print("  Hypothesis types:")
    for h_type, count in sorted(h_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {h_type.name}: {count}")

    # Run optimization (includes constraint matrix building)
    print("\nRunning optimization...")
    t_opt_start = time.perf_counter()
    optimised = tracker.optimise()
    t_opt = time.perf_counter() - t_opt_start
    print(f"  Optimization completed in {t_opt:.3f}s")
    print(f"  Selected {len(optimised)} hypotheses")
    print(f"  Final tracks: {tracker.n_tracks}")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Tracking time:          {t_track:.3f}s")
    print(f"  Hypothesis generation:  {t_hyp:.3f}s")
    print(f"  Optimization time:      {t_opt:.3f}s")
    print(f"  Total time:             {t_track + t_hyp + t_opt:.3f}s")
    print("=" * 60)

    return {
        "n_objects": n_objects,
        "n_frames": n_frames,
        "n_hypotheses": len(hypotheses),
        "n_tracks": tracker.n_tracks,
        "t_track": t_track,
        "t_hypothesis": t_hyp,
        "t_optimize": t_opt,
    }


if __name__ == "__main__":
    print("Hypothesis Generation Benchmark\n")

    # Run benchmark with different sizes
    test_sizes = [
        (50, 25),  # Small: 1,250 objects
        (100, 50),  # Medium: 5,000 objects
        (150, 75),  # Large: 11,250 objects
    ]

    results = []
    for n_obj, n_frames in test_sizes:
        print(f"\n{'=' * 60}")
        print(f"TEST: {n_obj} objects/frame × {n_frames} frames")
        print(f"{'=' * 60}\n")
        result = benchmark_hypothesis_generation(n_obj, n_frames)
        results.append(result)
        print("\n")

    # Summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Objects':<10} {'Frames':<10} {'Hypotheses':<12} {'Track(s)':<10} {'Hyp Gen(s)':<12} {'Optimize(s)':<12}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['n_objects']:<10} {r['n_frames']:<10} {r['n_hypotheses']:<12} "
            f"{r['t_track']:<10.3f} {r['t_hypothesis']:<12.3f} {r['t_optimize']:<12.3f}"
        )
    print("=" * 80)
