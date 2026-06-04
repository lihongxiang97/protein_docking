from docking.structure import (
    Atom,
    ProteinStructure,
    Residue,
    ensure_unique_chain_ids,
    merge_structures,
    split_complex_by_reference_chains,
)


def make_structure(name: str, chain_id: str, start_serial: int = 1) -> ProteinStructure:
    structure = ProteinStructure(name=name)
    atoms = [
        Atom(
            serial=start_serial,
            name="CA",
            alt_loc=" ",
            resname="ALA",
            chain_id=chain_id,
            resseq=1,
            icode=" ",
            x=1.0,
            y=2.0,
            z=3.0,
            element="C",
        ),
        Atom(
            serial=start_serial + 1,
            name="CB",
            alt_loc=" ",
            resname="ALA",
            chain_id=chain_id,
            resseq=1,
            icode=" ",
            x=1.5,
            y=2.5,
            z=3.5,
            element="C",
        ),
    ]
    residue = Residue(chain_id=chain_id, resseq=1, resname="ALA", icode=" ", atoms=atoms[:])
    structure.atoms.extend(atoms)
    structure.residues[(chain_id, 1, "ALA")] = residue
    structure.chains.add(chain_id)
    return structure


def test_ensure_unique_chain_ids_remaps_overlap():
    receptor = make_structure("receptor", "A")
    ligand = make_structure("ligand", "A", start_serial=10)

    remapped = ensure_unique_chain_ids(ligand, receptor.chains, name="ligand_unique")

    assert remapped.name == "ligand_unique"
    assert remapped.chains != ligand.chains
    assert remapped.chains.isdisjoint(receptor.chains)
    assert {atom.chain_id for atom in remapped.atoms} == remapped.chains


def test_split_complex_by_reference_chains_separates_structures():
    receptor = make_structure("receptor", "A")
    ligand = ensure_unique_chain_ids(make_structure("ligand", "A", start_serial=10), receptor.chains)
    complex_structure = merge_structures(receptor, ligand)

    receptor_part, ligand_part = split_complex_by_reference_chains(complex_structure, receptor.chains)

    assert receptor_part.chains == receptor.chains
    assert ligand_part.chains == ligand.chains
    assert len(receptor_part.atoms) == len(receptor.atoms)
    assert len(ligand_part.atoms) == len(ligand.atoms)
