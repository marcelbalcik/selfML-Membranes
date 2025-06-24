from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
import numpy as np

# Generate Morgan fingerprints for each molecule (circular fingerprints)
def get_morgan_fps(mols, radius=2, n_bits=2048):
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp))
    return np.array(fps)

# Generate MACCS keys (166-bit structural key fingerprint)
def get_maccs_fps(mols):
    fps = []
    for mol in mols:
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(np.array(fp))
    return np.array(fps)

# Generate topological (path-based) fingerprints
# This is similar to Daylight fingerprints
# Bit length can be adjusted, but 2048 is typical

def get_topological_fps(mols, n_bits=2048):
    fps = []
    for mol in mols:
        fp = RDKFingerprint(mol, fpSize=n_bits)
        fps.append(np.array(fp))
    return np.array(fps)

# Main dispatcher function that selects which fingerprint to compute
def get_fingerprints(smiles_list, method='morgan'):
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES encountered: {smi}")
        mols.append(mol)

    # Compute selected fingerprint type
    if method == 'morgan':
        return get_morgan_fps(mols)
    elif method == 'maccs':
        return get_maccs_fps(mols)
    elif method == 'topological':
        return get_topological_fps(mols)
    elif method == 'all':
        # Concatenate all three types horizontally into a composite descriptor
        morgan = get_morgan_fps(mols)
        maccs = get_maccs_fps(mols)
        topo = get_topological_fps(mols)
        return np.hstack((morgan, maccs, topo))
    else:
        raise ValueError(f"Unsupported fingerprint method: {method}")
