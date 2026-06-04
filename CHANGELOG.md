# Changelog

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
