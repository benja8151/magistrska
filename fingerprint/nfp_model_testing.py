import numpy as np
import pandas as pd

import tensorflow as tf
import nfp

batch_size = 512

# Load the input data, here YSI (10.1016/j.combustflame.2017.12.005)
ysi = pd.read_csv('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/data/qm9.csv')

# Split the data into training, validation, and test sets
valid, test, train = np.split(ysi.SMILES.sample(frac=1., random_state=1), [25000, 50000])

# Define how to featurize the input molecules
def atom_featurizer(atom):
    """ Return an string representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetIsAromatic(),
        nfp.get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
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

test_dataset = tf.data.Dataset.from_generator(
    lambda: ((preprocessor.construct_feature_matrices(row.SMILES, train=False), row.homo)
             for i, row in ysi[ysi.SMILES.isin(test)].iterrows()),
    output_types=(preprocessor.output_types, tf.float32),
    output_shapes=(preprocessor.output_shapes, []))\
    .cache()\
    .padded_batch(batch_size=batch_size, 
                  padded_shapes=(preprocessor.padded_shapes(), []),
                  padding_values=(preprocessor.padding_values, 0.))\
    .prefetch(tf.data.experimental.AUTOTUNE)

# Load saved model
model = tf.keras.models.load_model('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/model')

test_predictions = model.predict(test_dataset)

print(test_predictions[:5])
print(test_dataset.take(5))