# Reliable Protein Complex And PPI Benchmark Sources

This project should use experimentally supported data in three tiers:

1. **Gold-standard docking tests**: small to medium curated sets with bound and unbound
   structures, native complexes, categories, and known chain mappings.
2. **Large structure-derived ML data**: many PDB-derived complexes for learning interface
   or pose-ranking features, after redundancy filtering.
3. **Experimentally curated PPI pairs**: reliable positive interaction pairs that may need
   structure mapping before they can be used for docking.

## Recommended Sources

| Source | Scale | Best Use | Reliability Notes | URL |
|---|---:|---|---|---|
| Protein-Protein Docking Benchmark 5.5 (DB5.5) | 162 rigid-body cases plus medium/difficult and antibody-antigen cases in the official table | Primary blind/hybrid docking validation; LRMSD/iRMSD/FNAT/DockQ reporting | Curated bound and unbound structures with chain mappings and comments | https://zlab.wenglab.org/benchmark/benchmark5.5.html |
| DOCKGROUND experimental unbound sets | 92, 242, 233, and 396 complexes across sets 1-4 | Larger benchmark expansion and decoy-based scoring tests | Curated docking benchmarks from the Vakser Lab; includes unbound structures and decoys | https://dockground.compbio.ku.edu/ |
| DOCKGROUND docking decoys | 101 x 61 and 100 x 396 X-ray unbound decoy sets | ML pose-ranking and score calibration | Useful because incorrect and near-native poses are already supplied | https://dockground.compbio.ku.edu/ |
| DIPS-Plus | 42,112 complexes | Geometric/deep learning for interface prediction and feature pretraining | PDB-derived binary complexes with residue-level features and structure-based splits | https://pmc.ncbi.nlm.nih.gov/articles/PMC10400622/ |
| SKEMPI 2.0 | 7,085 mutations | Binding-energy and mutation-aware score calibration | Complex structures are solved and available in PDB; labels are thermodynamic/kinetic mutation effects | https://life.bsc.es/pid/skempi2/ |
| RCSB PDB Search/Data API | Custom scale | Fresh experimental biological assemblies, domain-specific holdout sets | Filter to experimental methods, low resolution cutoff, >=2 protein entities, biological assembly evidence | https://search.rcsb.org/ and https://data.rcsb.org/ |
| IntAct / IMEx | Millions of curated interaction records through IntAct/PSICQUIC | PPI classifier positives and candidate pairs for structure mapping | Use direct/physical interaction evidence and low-throughput filters when possible | https://www.ebi.ac.uk/intact |
| BioGRID | Large curated interaction database | PPI classifier positives; disease/pathway-specific candidates | Manually curated from primary literature; not all interactions imply a direct docking interface | https://thebiogrid.org/ |

## Practical Dataset Plan

### Phase 1: Trustworthy Docking Validation

Use DB5.5 first. Keep it independent from training.

```powershell
python scripts/collect_reliable_ppi_benchmarks.py db55 `
  --out data/reliable_ppi/db55 `
  --download-archive `
  --extract-archive `
  --download-table
```

Outputs:

- `manifest.json`: parsed DB5.5 bound complex IDs, receptor chains, ligand chains, categories.
- `manifest.csv`: spreadsheet-friendly manifest.
- `benchmark5.5.tgz`: official cleaned-up DB5.5 PDB archive from Weng lab.
- `Table_BM5.5.xlsx`: official case table.
- `archive/`: safely extracted official structures when `--extract-archive` is used.
- `pdb/`: downloaded RCSB PDB files when `--download-pdb` is additionally used.
- `pairs/`: receptor/ligand bound-chain split files when `--split-bound` is additionally used.

For final reporting, run docking in strict blind mode on this set and compute:

- Top-1 / Top-5 DockQ success
- CAPRI high / medium / acceptable counts
- LRMSD, iRMSD, FNAT distributions
- runtime per complex

### Phase 2: Pose-Ranking Machine Learning

Use DOCKGROUND decoys or self-generated decoys from DB5.5/DIPS-derived complexes.
Label each pose with DockQ using `docking.metrics.evaluate_complex`.

Recommended features:

- existing score components: hydrophobic, electrostatic, contacts, interface area,
  clash penalty, hydrogen bonds, restraints, prior/search score
- geometric features: contact residue count, interface area, cluster size
- optional residue-pair features: charged contacts, hydrophobic contacts, residue-pair
  frequency bins

Train a reranker:

```powershell
python scripts/build_pose_training_set.py `
  --manifest data/reliable_ppi/db55/manifest.json `
  --out results/benchmark/pose_training_rows.csv `
  --blind `
  --top-n 10 `
  --include-native-pose

python scripts/train_pose_reranker.py `
  --input results/benchmark/pose_training_rows.csv `
  --model-out models/default_pose_reranker.joblib `
  --model-type random_forest
```

Then enable it in `config.yaml`:

```yaml
docking:
  reranker_model: models/default_pose_reranker.joblib
  reranker_weight: 40.0
```

User-provided training manifests should use this CSV schema:

```csv
case_id,receptor_path,ligand_path,native_complex_path,receptor_chains,ligand_chains
case_001,data/user/case_001_rec.pdb,data/user/case_001_lig.pdb,data/user/case_001_native.pdb,A,B
```

The same `build_pose_training_set.py` command accepts this user CSV and emits the
same pose-feature rows used by `train_pose_reranker.py`.

### Phase 3: Large-Scale Interface Learning

Use DIPS-Plus for deep learning or residue-interface classifiers. This is better suited
for interface/contact prediction than direct blind docking accuracy. Keep DB5.5 and
DOCKGROUND held out to avoid leakage.

Recommended split hygiene:

- cluster by sequence identity or structural similarity,
- never put homologous complexes in both train and test,
- keep antibody-antigen, enzyme-inhibitor, and other classes stratified,
- deduplicate PDB entries and biological assemblies.

### Phase 4: Interaction-Pair Expansion

Use IntAct/IMEx or BioGRID for experimentally curated PPI positives. These pairs are not
automatically docking-ready; map proteins to high-quality structures first and label the
source evidence type. Prefer direct physical interaction evidence over association-only
records.

## Reliability Rules

- Prefer X-ray/cryo-EM/NMR PDB complexes with explicit chain mappings.
- Exclude homomers unless the algorithm is explicitly being tested for homomeric docking.
- Keep training and final benchmark PDB IDs disjoint.
- Record the source URL, download date, PDB ID, chain mapping, and filtering criteria.
- Use experimental PPI databases for interaction classification only unless complex
  structures or reliable mapped structures are available.
