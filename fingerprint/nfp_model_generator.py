import numpy as np
import pandas as pd

from rdkit import Chem
import tensorflow as tf
import nfp
import torch

def saveFingerprint(fingerprint, name, output_dir):
    tensor  = torch.tensor(fingerprint)
    torch.save(tensor, output_dir + "/" + name + ".pt")

# Load saved model
model = tf.keras.models.load_model('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/model')

# Load smiles
smiles_csv = pd.read_csv('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/smiles/smilesAll.csv')
print(smiles_csv.head())

# Prepare dataset
def atom_featurizer(atom):
    """ Return an string representing the atom type
    """

    # 10 params
    return str((
        atom.GetSymbol(),
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        nfp.get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True),
        atom.GetNumRadicalElectrons(),
        atom.GetMass(),
        Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
    ))

def bond_featurizer(bond, flipped=False):
    """ Get a similar classification of the bond type.
    Flipped indicates which 'direction' the bond edge is pointing. """
    
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()

preprocessor = nfp.SmilesPreprocessor(atom_features=atom_featurizer, bond_features=bond_featurizer,
                                      explicit_hs=False)

dataset = tf.data.Dataset.from_generator(
    lambda: (preprocessor.construct_feature_matrices(smiles, train=False)
             for smiles in smiles_csv.SMILES),
    output_types=preprocessor.output_types,
    output_shapes=preprocessor.output_shapes)\
    .padded_batch(batch_size=64, 
                  padded_shapes=preprocessor.padded_shapes(),
                  padding_values=preprocessor.padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

# Predicted fingerprints
fingerprints = model.predict(dataset)

for index, row in smiles_csv.iterrows():
    saveFingerprint(fingerprints[index], row['Filename'], 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/generated')