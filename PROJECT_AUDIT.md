# Project Audit Report

Date: 2026-06-04

## 1. Overall Architecture

This project is a Python protein-protein docking and PPI prediction workflow with a CLI entry point (`main.py`), a Streamlit UI (`web/`), core docking modules (`docking/`), data preparation/benchmark helpers (`tests/benchmark_data.py`, `scripts/`), and generated example/result data under `data/` and `results/`.

Core pipeline:

1. Parse PDB files into in-memory structure objects.
2. Preprocess structures by selecting chains, removing water/ions/HETATM, adding approximate hydrogens, estimating SASA, and assigning residue charges.
3. Generate rigid-body ligand poses with coarse rotations/translations and optional Monte Carlo refinement.
4. Score each receptor-ligand pose.
5. Rank poses and save docked complexes.
6. Analyze interface residues.
7. Predict PPI likelihood from docking/interface features.
8. Generate reports, plots, and optional 3D visualization.

## 2. Module Responsibilities

- `main.py`: CLI orchestration for docking, benchmark, and Streamlit launch.
- `docking/structure.py`: PDB parsing, atom/residue/structure data model, chain extraction/remapping, PDB writing.
- `docking/preprocess.py`: validation, cleanup, chain selection, rough hydrogen addition, SASA/charge assignment.
- `docking/surface.py`: approximate SASA and surface/patch detection.
- `docking/geometry.py`: rigid transforms, rotations, clash/contact/interface approximations.
- `docking/scoring.py`: multi-component docking score and pose ranking.
- `docking/interface.py`: residue-level interface detection, contact maps, network conversion.
- `docking/ppi_predictor.py`: rule-based or Random Forest PPI prediction.
- `docking/docking.py`: end-to-end docking search and pose persistence.
- `docking/visualization.py`: static plots, py3Dmol HTML, ECharts JSON generation.
- `web/app.py`, `web/structure_viewer.py`: Streamlit UI and NGL-based 3D viewer.
- `tests/benchmark_data.py`: example and benchmark PDB download/extraction/synthesis.
- `tests/benchmark_test.py`, `tests/evaluation.py`: benchmark execution and report generation.

## 3. Data Flow And Calls

CLI docking:

`main.py -> StructurePreprocessor.validate_structure -> ProteinDocker.dock -> StructurePreprocessor.preprocess -> SurfaceAnalyzer -> ProteinDocker coarse/MC search -> DockingScorer.score_complex -> merge_structures/write_pdb -> InterfaceAnalyzer.analyze -> PPIPredictor.predict -> ResultVisualizer/report writers`

Web docking:

`web/app.py -> file upload/example generation -> ProteinDocker.dock -> InterfaceAnalyzer/PPIPredictor per pose -> ResultVisualizer ECharts JSON -> NGL viewer payload`

Benchmark:

`BenchmarkRunner.run_all -> prepare_benchmark_dataset -> ProteinDocker.dock -> InterfaceAnalyzer -> PPIPredictor -> simplified metrics -> EvaluationReport plots/report`

## 4. Design Defects

- The docking search is not interface-aware. It initially translates the ligand to receptor center plus a fixed `+20 A` X offset, then samples a small cubic grid around that offset. This can miss most valid interfaces.
- Pose deduplication compares only total score, so distinct conformations with similar scores are incorrectly discarded while geometric duplicates with different scores can remain.
- The PPI prediction model is a thresholded rule stack with no calibration, no feature provenance, and no explicit quality gates for clash/contact contradictions.
- Scoring uses raw CA/residue approximations for several biological effects. It is useful for demonstration, but the implementation currently presents some outputs as stronger than the method supports.
- Benchmark RMSD/FNAT/DockQ are placeholder approximations and are not comparable to CAPRI/DockQ standards.
- Web UI and README contain mojibake text, reducing usability and maintainability.

## 5. Confirmed Bugs

- Runtime import fails in the current environment because `scipy` is missing but `cKDTree` is imported unconditionally.
- `pytest` is missing, so test execution via `python -m pytest` is unavailable in the current environment.
- `main.py --top-n` is parsed but never applied to `ProteinDocker.top_n`.
- Configuration loading does not validate YAML type or numeric ranges. Bad config values can silently create invalid searches.
- `Residue.key` and `Atom.residue_key` omit insertion code, merging residues that share chain/residue number/residue name but differ by insertion code.
- `SurfaceAnalyzer.compute_sasa` aggregates atom SASA by assuming `structure.atoms` order matches sorted residues. After parsing or rebuilding, this can assign atom SASA to the wrong residue.
- `PDBParser` does not handle alternate locations consistently; duplicate alt-loc atoms can inflate contacts/clashes.
- `_generate_translations` treats `coarse_translations` as both radius and result count, then truncates the generated grid to `n**3+1`, producing a non-obvious subset.
- `count_clashes` accepts `clash_cutoff` but ignores it in the actual clash decision.
- Monte Carlo acceptance uses `exp(delta / temperature)` without clipping; extreme values can overflow if scoring changes.
- `DockingScorer._score_electrostatic` relies on residue charges being pre-assigned. If scoring receives parsed-but-not-preprocessed structures, electrostatics are always zero.
- `ResultVisualizer.generate_3d_view` can fail if the PDB path is missing or py3Dmol is unavailable; only import failure is handled.
- Web/HTML templates contain malformed text and at least one broken HTML label fragment, which can affect rendering.
- Benchmark runner skips missing ligand files less explicitly than receptor files and logs errors without structured recovery metadata.

## 6. Performance Bottlenecks

- Each pose builds copied ligand structures and merged complexes before ranking. This causes large memory churn.
- Scoring repeatedly builds KD-trees per pose and per component.
- Coarse search is single-threaded and has no early stopping other than clash pruning.
- Static plot generation can be expensive and is invoked even for empty or low-information outputs.
- Web pose detail computation re-analyzes every returned pose synchronously.

## 7. Code Quality And Maintainability Issues

- Most Chinese comments/strings are mojibake in the repository, making docs and UI hard to maintain.
- Imports in `docking/__init__.py` eagerly import optional visualization and SciPy-dependent modules, making lightweight structure utilities fail when optional dependencies are absent.
- Error handling is uneven: some modules return empty outputs, some raise, and some log-and-continue.
- There is no central config schema/default normalization.
- Generated artifacts (`__pycache__`, results) are present in the working tree.
- Tests cover only chain remapping/splitting and do not cover docking, scoring, parsing edge cases, config, or no-SciPy fallback behavior.

## 8. Numeric Stability And Scientific Validity Risks

- Clash/contact thresholds mix atom-level and CA-level assumptions.
- Hydrogen bonds are distance-only and ignore donor/acceptor directionality.
- SASA is a rough neighbor-overlap estimate, not a Shrake-Rupley implementation.
- Interface area is approximated from contacting atoms times a fixed area.
- Docking score mixes normalized and clipped features but PPI rules compare raw thresholds, so interpretation can drift.

## 9. Immediate Fix Plan

1. Add a lightweight spatial index abstraction with SciPy fallback to NumPy brute force.
2. Harden PDB parsing, residue keys, config validation, input checks, and output writing.
3. Make docking search interface-aware through surface/contact anchor sampling while preserving public APIs.
4. Replace score-only pose deduplication with geometric deduplication.
5. Make PPI prediction more explainable via named feature contributions and quality gates.
6. Add targeted tests that run without pytest using `unittest`, while still remaining pytest-compatible.
7. Generate a detailed `CHANGELOG.md`.
