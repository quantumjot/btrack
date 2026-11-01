# btrack Rust Porting Plan

This document outlines the architecture, roadmap, and implementation strategy for porting the `btrack` C++ tracking engine and Python ILP solver (`cvxopt.glpk`) to a unified, high-performance Rust backend.

---

## 1. Executive Summary & Goals

### Current Limitations
1. **Fragile C++ Compilation & Distribution**:
   - Relies on custom `Makefile` and `build.sh` scripts that dynamically clone Eigen 3.3.9 into `btrack/include/eigen`.
   - Generates a platform-dependent dynamic library (`libtracker.dylib`, `libtracker.so`, `tracker.DLL`) loaded via `ctypes` (`btrack/libwrapper.py`). Missing compiled binaries cause runtime errors rather than build-time feedback.
   - Cross-compiling C++ for various OS/architecture targets (macOS arm64/x86_64, Linux x86_64/aarch64, Windows) is brittle and complex.
2. **Heavy External Optimization Dependencies (`cvxopt`)**:
   - `btrack` relies on `cvxopt` (and GLPK under the hood) to solve the Integer Linear Programming (ILP) association problem.
   - `cvxopt` requires compiled C/Fortran math libraries, complicating installation across newer Python versions and platforms.
3. **Data Marshaling Overhead**:
   - Hypotheses generated in C++ are passed back through `ctypes` into Python objects, converted into sparse matrices via `cvxopt.spmatrix`, solved in GLPK, and sent back across the boundary into C++ for merging.

### Target Architecture
* **Single Rust Extension (`btrack_core`)**:
  - Implements the complete tracking pipeline: Kalman filtering, Bayesian belief calculation, spatial hashing, hypothesis generation, ILP formulation and solving, and track merging.
  - Built with **`maturin`** and **`pyo3`** (`rust-numpy`), producing standard self-contained Python wheels.
  - Eliminates `cvxopt`, `Makefile`, `build.sh`, Eigen download scripts, and `ctypes` pointer gymnastics.
  - Native embedded Mixed-Integer Linear Programming (MILP) solver via **HiGHS** (or `good_lp`).

```
┌────────────────────────────────────────────────────────┐
│ Python Interface (btrack.BayesianTracker)              │
│  - User configuration & datasets                       │
│  - napari plugins & export utilities                   │
└──────────────────────────┬─────────────────────────────┘
                           │ PyO3 / NumPy (zero-copy)
┌──────────────────────────▼─────────────────────────────┐
│ Rust Core Engine (btrack_core)                         │
│  ├── Kalman Filter & Motion Models (nalgebra)          │
│  ├── Bayesian Belief Updates & Likelihoods (statrs)   │
│  ├── Spatial-Temporal Hashing (HypercubeBin)           │
│  ├── Hypothesis Generation Engine                      │
│  ├── Built-in ILP / MIP Solver (HiGHS / good_lp)       │
│  └── Tracklet, Manager & Graph Lineages                │
└────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack & Crate Selection

| Component | Current Stack | Rust Replacement | Justification |
| :--- | :--- | :--- | :--- |
| **Python Bindings** | `ctypes` (`libwrapper.py`) | `pyo3` (v0.22+) + `numpy` | Zero-overhead ABI, safe memory/GIL management, typed Python exceptions. |
| **Build Backend** | `setuptools` + `Makefile` | `maturin` | Modern PyPA standard for Rust-Python extensions; seamless `pip install -e .` and cross-compilation. |
| **Linear Algebra** | `Eigen` 3.3.9 | `nalgebra` | High-performance, pure Rust, stack-allocated fixed matrices (e.g. `SMatrix<f64, 6, 6>`) for Kalman filters. |
| **Distributions / Math** | `std::erf`, Gaussian PDF | `statrs` / `libm` | Numerical accuracy for erf integrals and normal distributions. |
| **ILP Solver** | `cvxopt.glpk` | `highs` or `good_lp` (HiGHS backend) | HiGHS is the premier open-source LP/MIP solver (MIT licensed, used by SciPy), fast, reliable, and portable. |
| **Spatial Indexing** | `HypercubeBin` / `ObjectBin` | Idiomatic Rust hash-bins or `rstar` | $O(1)$ amortized neighbor searches for track linkage and hypothesis candidate generation. |

---

## 3. Implementation Roadmap

### Phase 1: Workspace & Build System Setup
- [x] Initialize Cargo project structure under `crates/btrack_core` (or root `Cargo.toml`).
- [x] Configure `pyproject.toml` with `maturin` as the build backend.
- [x] Set up basic PyO3 skeleton module to verify compiling and importing inside the Python virtual environment.
- [x] Ensure local dev workflow works: `maturin develop` inside the python venv.

### Phase 2: Core Data Types & Mathematical Primitives
*Source references: `btrack/include/types.h`, `defs.h`, `pdf.cc`, `bayes.cc`, `motion.cc`, `hyperbin.cc`*

- [x] **TrackObject & ImagingVolume**:
  - `TrackObject`: Position $(x, y, z)$, frame $t$, dummy flag, label, and feature vectors.
  - `ImagingVolume`: Bounding volume, automatic 2D vs 3D detection, boundary proximity calculation (`dist_from_border`).
- [x] **MotionModel (Kalman Filter)**:
  - Transition matrix $A$, observation matrix $H$, covariance $P$, process noise $Q$, measurement noise $R$.
  - Prediction and update steps:
    $$\hat{x}_{k|k-1} = A \hat{x}_{k-1}, \quad P_{k|k-1} = A P_{k-1} A^T + Q$$
    $$K = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$
    $$\hat{x}_k = \hat{x}_{k|k-1} + K (z_k - H \hat{x}_{k|k-1}), \quad P_k = (I - KH) P_{k|k-1}$$
- [x] **Bayesian Updating & Error Function PDF**:
  - `multivariate_erf`: Exact erf-based spatial overlap integral over observation bounding regions.
  - `safe_bayesian_update`: Bayesian odds calculation with numerical clamping.
- [x] **Spatial-Temporal Hashing (`HypercubeBin`)**:
  - Multi-dimensional grid binning for localized neighbor lookups.

### Phase 3: Tracklet Management & Tracker Engine
*Source references: `tracklet.cc`, `manager.cc`, `tracker.cc`*

- [x] **Tracklet**:
  - Store observation histories, Kalman states $(\mu, \Sigma)$, track status (`active`, `lost`), lineage IDs (`parent`, `root`, `children`), and track fate.
- [x] **TrackManager**:
  - ID generation, track storage, active/lost tracking lists, dummy point insertion, and candidate graph edge tracking.
- [x] **BayesianTracker**:
  - Frame-by-frame loop (`step()` and `track()`):
    1. Filter active tracks within volume.
    2. Collect observations for current frame.
    3. Construct belief matrix (`EXACT` and `APPROXIMATE` modes).
    4. Resolve associations, handle conflict resolution, append linked observations, or insert dummy predictions for missing frames.

### Phase 4: Hypothesis Generation Engine
*Source references: `btrack/include/hypothesis.h`, `hypothesis.cc`*

- [ ] Define hypothesis types:
  - `FalsePositive` ($P_{FP}$)
  - `Initialize` ($P_{init}$)
  - `Terminate` ($P_{term}$)
  - `Link` ($P_{link}$)
  - `Branch` / Division ($P_{branch}$)
  - `Apoptosis` / Dead ($P_{dead}$)
  - `Merge` ($P_{merge}$)
- [ ] Implement `HypothesisEngine`:
  - Query spatial hash bins for candidate track pairs/triplets within time threshold $\Delta t \le \theta_{time}$ and distance threshold $d \le \theta_{dist}$.
  - Compute log-likelihood penalties with user parameters ($\lambda_{time}, \lambda_{dist}, \lambda_{link}, \lambda_{branch}$, etc.).

### Phase 5: Rust ILP Solver (Replacing `cvxopt.glpk`)
*Source reference: `btrack/optimise/optimiser.py`*

- [ ] **Problem Formulation**:
  - Decision variables $x_k \in \{0, 1\}$ for each hypothesis $k \in \{0, \dots, M-1\}$.
  - Objective:
    $$\min \sum_{k=0}^{M-1} -\rho_k x_k \quad \text{where } \rho_k = \ln P(h_k)$$
  - Subject to exact inflow and outflow constraints:
    $$A x = \mathbf{1}_{2N}$$
    where $N$ is the number of tracklets, row $trk$ represents outflow/termination, and row $N + trk$ represents inflow/initialization.
- [ ] **HiGHS Integration**:
  - Construct sparse constraints directly in memory using the `highs` crate (or `good_lp`).
  - Solve the binary integer program.
  - Return the indices of selected hypotheses ($x_k = 1$).
  - Remove `cvxopt` from Python dependencies!

### Phase 6: Post-Optimization Graph Merging & Lineage
*Source reference: `btrack/src/manager.cc`*

- [ ] Execute selected hypotheses:
  - Splice linked tracklets into single trajectories.
  - Set lineage parent/child references for cell division branches.
  - Handle apoptosis terminations and false positive removals.
  - Renumber tracks and compute generational depth.

### Phase 7: Python API Integration & Backward Compatibility
- [ ] Bind the Rust tracker to Python using PyO3.
- [ ] Provide 100% backward-compatible methods in `btrack.BayesianTracker`:
  - `.append(objects)`
  - `.track(step_size=...)`
  - `.hypotheses()`
  - `.optimise(options=...)`
  - `.tracks`, `.refs`, `.dummies`, `.to_napari()`
- [ ] Remove `btrack/libwrapper.py`, C++ source files (`btrack/src/`, `btrack/include/`), `Makefile`, and `build.sh`.

### Phase 8: Verification, Benchmarking & CI
- [ ] **Correctness**: Validate against `tests/_test_data/test_ground_truth.json` and `test_graph.json`.
- [ ] **Numerical Consistency**: Assert Kalman trajectories match within $10^{-6}$ tolerance.
- [ ] **Performance Benchmarks**: Compare execution times across tracking and optimization phases against the C++/GLPK baseline.
- [ ] **CI Wheels**: Setup `cibuildwheel` / `maturin-action` in GitHub Actions for macOS (arm64, x86_64), Linux (x86_64, aarch64), and Windows (x86_64).
