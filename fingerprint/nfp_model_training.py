import numpy as np
import pandas as pd

import tensorflow as tf
import nfp

print(f"tensorflow {tf.__version__}")
print(f"nfp {nfp.__version__}")

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

# Initially, the preprocessor has no data on atom types, so we have to loop over the 
# training set once to pre-allocate these mappings
for smiles in train:
    preprocessor.construct_feature_matrices(smiles, train=True)

# Construct the tf.data pipeline. There's a lot of specifying data types and
# expected shapes for tensorflow to pre-allocate the necessary arrays. But 
# essentially, this is responsible for calling the input constructor, batching 
# together multiple molecules, and padding the resulting molecules so that all
# molecules in the same batch have the same number of atoms (we pad with zeros,
# hence why the atom and bond types above start with 1 as the unknown class)
train_dataset = tf.data.Dataset.from_generator(
    lambda: ((preprocessor.construct_feature_matrices(row.SMILES, train=False), row.homo)
             for i, row in ysi[ysi.SMILES.isin(train)].iterrows()),
    output_types=(preprocessor.output_types, tf.float32),
    output_shapes=(preprocessor.output_shapes, []))\
    .cache().shuffle(buffer_size=200)\
    .padded_batch(batch_size=batch_size, 
                  padded_shapes=(preprocessor.padded_shapes(), []),
                  padding_values=(preprocessor.padding_values, 0.))\
    .prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(
    lambda: ((preprocessor.construct_feature_matrices(row.SMILES, train=False), row.homo)
             for i, row in ysi[ysi.SMILES.isin(valid)].iterrows()),
    output_types=(preprocessor.output_types, tf.float32),
    output_shapes=(preprocessor.output_shapes, []))\
    .cache()\
    .padded_batch(batch_size=batch_size, 
                  padded_shapes=(preprocessor.padded_shapes(), []),
                  padding_values=(preprocessor.padding_values, 0.))\
    .prefetch(tf.data.experimental.AUTOTUNE)

## Define the keras model
from tensorflow.keras import layers

# Input layers
atom = layers.Input(shape=[None], dtype=tf.int64, name='atom')
bond = layers.Input(shape=[None], dtype=tf.int64, name='bond')
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

num_features = 8  # Controls the size of the model

# Convert from a single integer defining the atom state to a vector
# of weights associated with that class
atom_state = layers.Embedding(preprocessor.atom_classes, num_features,
                              name='atom_embedding', mask_zero=True)(atom)

# Ditto with the bond state
bond_state = layers.Embedding(preprocessor.bond_classes, num_features,
                              name='bond_embedding', mask_zero=True)(bond)

# Here we use our first nfp layer. This is an attention layer that looks at
# the atom and bond states and reduces them to a single, graph-level vector. 
# mum_heads * units has to be the same dimension as the atom / bond dimension
global_state = nfp.GlobalUpdate(units=8, num_heads=1)([atom_state, bond_state, connectivity])

for _ in range(3):  # Do the message passing
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([bond_state, new_bond_state])
    
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([atom_state, new_atom_state])
    
    new_global_state = nfp.GlobalUpdate(units=8, num_heads=1)(
        [atom_state, bond_state, connectivity, global_state]) 
    global_state = layers.Add()([global_state, new_global_state])
    
    
# Since the final prediction is a single, molecule-level property (YSI), we 
# reduce the last global state to a single prediction.
fp_out = layers.Dense(num_features)(global_state)
ysi_prediction = layers.Dense(1)(global_state)

# Construct the tf.keras models - one with YSI output, the other with FP
model_ysi = tf.keras.Model([atom, bond, connectivity], [ysi_prediction])
model_fp = tf.keras.Model([atom, bond, connectivity], [fp_out])

model_ysi.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(1E-3))
model_fp.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(1E-3))

# Fit the model - only fit ysi model. The first epoch is slower, since it needs to cache
# the preprocessed molecule inputs
model_ysi.fit(train_dataset, validation_data=valid_dataset, epochs=25)

# Here, we create a test dataset that doesn't assume we know the values for the YSI
test_dataset = tf.data.Dataset.from_generator(
    lambda: (preprocessor.construct_feature_matrices(smiles, train=False)
             for smiles in test),
    output_types=preprocessor.output_types,
    output_shapes=preprocessor.output_shapes)\
    .padded_batch(batch_size=batch_size, 
                  padded_shapes=preprocessor.padded_shapes(),
                  padding_values=preprocessor.padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

# Here are the predictions on the test set
test_predictions = model_ysi.predict(test_dataset)
#test_db_values = test_dataset.homo.values
test_db_values = ysi.homo.values
#print(ysi.set_index('SMILES').reindex(test))

test_db_reindexed_values = []

for index, prediction in test.items():
    test_db_reindexed_values.append(test_db_values[index])

print(np.abs(test_db_reindexed_values - test_predictions.flatten()).mean())
#print(tf.keras.losses.mean_absolute_percentage_error(test_db_values, test_predictions.flatten()))

# Save model
model_fp.save('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/model')