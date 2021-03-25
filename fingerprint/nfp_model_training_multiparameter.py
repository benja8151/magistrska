import numpy as np
import pandas as pd

import tensorflow as tf
import nfp
from rdkit import Chem

print(f"tensorflow {tf.__version__}")
print(f"nfp {nfp.__version__}")

batch_size = 512
num_features = 64  # Controls the size of the model
fp_size = 8
units = 64
heads = 1
epochs = 20
test_train_split = [25000, 50000]
loss_function = 'mae'

n_outputs = 3 #A, B, C

# Load the input data
csv_data = pd.read_csv('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/data/qm9.csv')

# Split the data into training, validation, and test sets
valid, test, train = np.split(csv_data.SMILES.sample(frac=1), test_train_split)

print(len(valid), len(test), len(train))

# Define how to featurize the input molecules
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
    isAromatic = str(int(bond.GetIsAromatic()))
    isConjugated = str(int(bond.GetIsConjugated()))
    ring = 'R{}'.format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring, isAromatic, isConjugated]).strip()

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
    lambda: ((preprocessor.construct_feature_matrices(row.SMILES, train=False), (row.A, row.B, row.C))
             for i, row in csv_data[csv_data.SMILES.isin(train)].iterrows()),
    output_types=(preprocessor.output_types, (tf.float32, tf.float32, tf.float32)),
    output_shapes=(preprocessor.output_shapes, ([], [], [],)))\
    .cache().shuffle(buffer_size=200)\
    .padded_batch(batch_size=batch_size, 
                  padded_shapes=(preprocessor.padded_shapes(), ([], [], [],)),
                  padding_values=(preprocessor.padding_values, (0., 0., 0.)))\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_generator(
    lambda: ((preprocessor.construct_feature_matrices(row.SMILES, train=False), (row.A, row.B, row.C))
             for i, row in csv_data[csv_data.SMILES.isin(valid)].iterrows()),
    output_types=(preprocessor.output_types, (tf.float32, tf.float32, tf.float32)),
    output_shapes=(preprocessor.output_shapes, ([], [], [],)))\
    .cache()\
    .padded_batch(batch_size=batch_size, 
                  padded_shapes=(preprocessor.padded_shapes(), ([], [], [],)),
                  padding_values=(preprocessor.padding_values, (0., 0., 0.)))\
    .prefetch(tf.data.experimental.AUTOTUNE)

## Define the keras model
from tensorflow.keras import layers

# Input layers
atom = layers.Input(shape=[None], dtype=tf.int64, name='atom')
bond = layers.Input(shape=[None], dtype=tf.int64, name='bond')
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

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
global_state = nfp.GlobalUpdate(units=units, num_heads=heads)([atom_state, bond_state, connectivity])

for _ in range(3):  # Do the message passing
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([bond_state, new_bond_state])
    
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([atom_state, new_atom_state])
    
    new_global_state = nfp.GlobalUpdate(units=units, num_heads=heads)(
        [atom_state, bond_state, connectivity, global_state]) 
    global_state = layers.Add()([global_state, new_global_state])
    
    
# Since the final prediction is a single, molecule-level property (YSI), we 
# reduce the last global state to a single prediction.
fp_out = layers.Dense(fp_size)(global_state)
param_prediction = layers.Dense(n_outputs)(global_state)

# Construct the tf.keras models - one with YSI output, the other with FP
model_param_prediction = tf.keras.Model([atom, bond, connectivity], [param_prediction])
model_fp = tf.keras.Model([atom, bond, connectivity], [fp_out])

model_param_prediction.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(1E-3))
model_fp.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(1E-3))

# Fit the model - only fit ysi model. The first epoch is slower, since it needs to cache
# the preprocessed molecule inputs
model_param_prediction.fit(train_dataset, validation_data=valid_dataset, epochs=epochs)


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


""" all_data = csv_data.SMILES.sample(frac=1., random_state=1)
all_dataset = tf.data.Dataset.from_generator(
    lambda: (preprocessor.construct_feature_matrices(smiles, train=False)
             for smiles in all_data),
    output_types=preprocessor.output_types,
    output_shapes=preprocessor.output_shapes)\
    .padded_batch(batch_size=batch_size, 
                  padded_shapes=preprocessor.padded_shapes(),
                  padding_values=preprocessor.padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE) """


# Here are the predictions on the test set
test_predictions = model_param_prediction.predict(test_dataset)
#test_db_values = test_dataset.homo.values
test_db_values = [csv_data.A.values, csv_data.B.values, csv_data.C.values] 
#print(ysi.set_index('SMILES').reindex(test))

""" 
all_predictions = model_param_prediction.predict(all_dataset)
all_reindexed_values = []
for index, prediction in all_data.items():
    all_reindexed_values.append(test_db_values[index])
print(np.round(np.divide(np.abs(all_predictions.flatten() - all_reindexed_values), all_reindexed_values) * 100)) """

test_db_reindexed_values = []

for index, prediction in test.items():
    test_db_reindexed_values.append([test_db_values[0][index], test_db_values[1][index], test_db_values[2][index]])

#print(list(zip(test_predictions, test_db_reindexed_values)))


#print(test_db_reindexed_values - test_predictions)
#print(np.abs(test_db_reindexed_values - test_predictions.flatten()).mean())
#print(tf.keras.losses.mean_absolute_percentage_error(test_db_values, test_predictions.flatten()))


# Save model
model_fp.save('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/model')