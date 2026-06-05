# Changelog

## 2026-06-05 - Benchmark Calibration And Report Cleanup

### Fixed

- Fixed saturated rule-based PPI benchmark predictions by adding an overpacked-interface penalty and reducing overly aggressive positive evidence weights.
- Fixed benchmark report portability by replacing non-ASCII multiplication symbols in complexity text with ASCII-safe `x`.
- Fixed benchmark report rendering in the Streamlit app by reading generated Markdown and PDB text with explicit UTF-8 decoding and replacement for malformed bytes.
- Fixed noisy benchmark plot generation on Windows by using matplotlib's bundled `DejaVu Sans` font instead of unavailable Helvetica fallbacks.

### Added

- Added a threshold diagnostic section to benchmark reports to distinguish fixed-threshold performance from split-local calibration behavior.
- Added regression tests for overpacked-interface probability penalties and ASCII-safe benchmark report output.

### Validation

- Recomputed the existing 23-case benchmark report after recalibration: accuracy `0.8696`, precision `0.9286`, recall `0.8667`, F1 `0.8966`, MCC `0.7238`, AUROC `0.9583`, AUPRC `0.9759`.
- Verified the regenerated report contains no non-ASCII characters or illegal control characters.
- Full regression suite passes: `18` tests.

## 2026-06-05 - Reliable Benchmark Data And ML Reranking

### Added

- Added `BENCHMARK_DATASETS.md` with reliable experimental protein-complex and PPI data sources for validation and training.
- Added `scripts/collect_reliable_ppi_benchmarks.py` to parse DB5.5, write manifests, download bound PDB files, and split receptor/ligand chains.
- Added optional `docking.ml_reranker.PoseReranker` and `scripts/train_pose_reranker.py` for DockQ/LRMSD-supervised pose reranking.

### Changed

- Added optional `docking.reranker_model` and `docking.reranker_weight` config keys. The existing score-based ranking remains the default.

## 2026-06-05 - CAPRI/DockQ Evaluation Metrics

### Added

- Added CAPRI/DockQ-style docking evaluation with heavy-atom FNAT, receptor-aligned LRMSD, interface iRMSD, DockQ, and CAPRI quality class.
- Added benchmark output fields for `lrmsd`, `irmsd`, `dockq`, and `capri_class` while keeping `rmsd` as the LRMSD compatibility column.
- Added regression tests for heavy-atom native contacts and shifted-ligand metric penalties.

### Changed

- Replaced the previous CA-contact simplification in `docking/metrics.py` with heavy-atom residue contact detection.
- Updated benchmark evaluation reports to show LRMSD, iRMSD, FNAT, DockQ, and CAPRI class.

## 2026-06-04 - Web Viewer Layout And Localization

### Added

- Added a top-right Chinese/English switch for the full Streamlit page.
- Added synchronized localization for page controls, result tables, PPI explanations, ECharts labels, and the embedded NGL viewer.
- Added viewer layout and localization regression tests.

### Fixed

- Fixed shortest-contact table rows visually scrolling over the sticky header.
- Removed the unintended gap between the shortest-contact panel heading and its table header.
- Isolated the contact-table stacking context and changed the header to a fully opaque, high-priority sticky layer.
- Kept the shortest-contact table bounded within the analysis panel.

## 2026-06-04 - Hybrid FFT Docking Architecture

### Added

- Added `docking/fft_search.py` with independent FFT-based global translation search.
- Added shape, electrostatic, and hydrophobic grid correlations.
- Added multi-seed local refinement and ligand-RMSD pose clustering.
- Added optional ambiguous active-residue restraints inspired by information-driven docking.
- Added receptor-aligned LRMSD, interface RMSD, native-contact fraction, and DockQ evaluation helpers.
- Added hybrid input-pose prior and strict blind-docking mode.
- Added `ALGORITHM_DESIGN.md` documenting method mapping, usage, limitations, and validation.
- Added CLI controls: `--blind`, `--rotations`, `--mc-iterations`, active residue lists, and restraint distances.

### Changed

- Replaced directional-only global search with FFT search by default.
- Added low-discrepancy rotation sampling.
- Strengthened clash-aware reranking and capped hydrogen-bond bonuses.
- Added `search_score`, `cluster_size`, `provenance`, restraint fields, and prior score to result CSV output.
- Fixed loss of residue charge/SASA/surface annotations during chain remapping, structure copying, and extraction.

### Validation

- Seven core regression tests pass.
- Hybrid mode preserves and reranks meaningful input relative poses; this prior must not be reported as blind-docking accuracy.
- In strict blind mode, FFT generated a 2.62 A near-native candidate, while current reranking placed the best Top-2 pose at 13.35 A; reranking remains the main accuracy bottleneck.

## 2026-06-04 - Audit, Bug Fixes, And Docking/PPI Refactor

### Added

- Added `PROJECT_AUDIT.md` with architecture, module, data-flow, bug, performance, and maintainability findings.
- Added `docking/spatial.py`, a SciPy-compatible `cKDTree` wrapper that falls back to a NumPy implementation when SciPy is unavailable.
- Added `docking/config.py` for centralized config loading and range/type validation.
- Added `tests/test_core_regressions.py` with standard-library `unittest` coverage for:
  - insertion-code-safe residue keys,
  - SASA aggregation with out-of-order atoms/residues,
  - spatial-index API compatibility,
  - invalid config rejection,
  - small end-to-end docking.

### Fixed

- Fixed runtime import failure when SciPy is missing by routing KD-tree usage through `docking.spatial`.
- Fixed runtime import failure when scikit-learn is missing by keeping Random Forest training optional and falling back to the rule-based PPI predictor.
- Fixed package-level eager imports in `docking/__init__.py` so importing lightweight modules no longer loads optional heavy dependencies.
- Fixed residue identity collision by including insertion code in atom/residue keys.
- Fixed SASA residue aggregation so it maps atom SASA by residue key instead of assuming atom order matches sorted residue order.
- Fixed alternate-location parsing by ignoring non-primary alt-loc atoms.
- Fixed `main.py --top-n` behavior by applying it to the docker before docking.
- Fixed Windows output failures by writing text reports and generated files with explicit UTF-8 encoding.
- Fixed numeric instability in Monte Carlo acceptance by clipping the exponential argument.
- Fixed clash detection so `clash_cutoff` affects the actual clash criterion.

### Refactored

- Reworked docking placement generation from a fixed `+X` offset and truncated cubic grid to approximate surface-contact sampling over multiple sphere directions.
- Reduced docking memory churn by storing full merged complex structures only for final ranked poses instead of every candidate pose.
- Replaced score-only pose deduplication with geometric ligand-coordinate deduplication.
- Kept existing public APIs for `ProteinDocker.dock`, `DockingScorer`, `InterfaceAnalyzer`, and `PPIPredictor`.

### Known Remaining Risks

- In the current environment, SciPy is not installed, so the NumPy fallback is correct but slow for real structures.
- Benchmark RMSD/FNAT/DockQ remain simplified placeholders and should not be interpreted as CAPRI/DockQ-standard metrics.
- The README may still need a documentation text cleanup pass.
- Web visualization depends on CDN-hosted ECharts/NGL, which may fail in offline environments.
