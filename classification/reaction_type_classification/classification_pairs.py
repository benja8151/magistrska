import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from confusion_matrix import make_confusion_matrix
from classification_mosaic import nclass_classification_mosaic_plot
from itertools import cycle, combinations

# Input Parameters
reactions_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions'
fingerprints_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/fingerprint_generator_new/generated'
reactions_csv = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/reaction_type_classification/data/reactions_all.csv'

n_types = 8
fp_length = 128
fp_multiplying_factor = 1
epochs = 50
batch_size = 25
learning_rate = 0.001
k_folds = 5

reaction_types = [
    "Hydrolase",
    "Isomerase",
    "Ligase",
    "Lyase",
    "Oxidoreductase",
    "Transferase",
    "Translocase",
    "Unassigned",
]

reaction_pairs_compare = [
    (4, 5), #oxi-trans,
    (5, 0), #hydro-trans,
    (4, 3), #oxi-lyase,
    (0, 2), #hydro-ligase,
    (3, 1), #lyase-iso,
    (2, 1), #ligase-iso
]
reaction_pairs_all = list(combinations(np.arange(len(reaction_types)), 2))

# Fingerprint Functions
def createCombinedFingerprint(reaction, reactions_dir, fingerprints_dir, combine_method="average"):
    '''
    Arguments
    ---------
    combine_method: "average" or "sum"
    '''
    try: 
        reaction_file = open(reactions_dir + "/" + reaction, 'r')
        
        reaction_split = reaction_file.read().split('-')
        left_side = reaction_split[0].split(',')
        right_side = reaction_split[1].split(',')

        if (combine_method == "average"):
            right_side_average = torch.load(fingerprints_dir + '/' + right_side[0] + '.pt').numpy()
            left_side_average = torch.load(fingerprints_dir + '/' + left_side[0] + '.pt').numpy()
            
            for i in range(1, len(right_side)):
                right_side_average += torch.load(fingerprints_dir + '/' + right_side[i] + '.pt').numpy()
            for i in range(1, len(left_side)):
                left_side_average += torch.load(fingerprints_dir + '/' + left_side[i] + '.pt').numpy()

            right_side_average /= len(right_side)
            left_side_average /= len(left_side)

            return right_side_average - left_side_average
        
        elif (combine_method == "sum"):
            right_side_sum = torch.load(fingerprints_dir + '/' + right_side[0] + '.pt').numpy()
            left_side_sum = torch.load(fingerprints_dir + '/' + left_side[0] + '.pt').numpy()
            
            for i in range(1, len(right_side)):
                right_side_sum += torch.load(fingerprints_dir + '/' + right_side[i] + '.pt').numpy()
            for i in range(1, len(left_side)):
                left_side_sum += torch.load(fingerprints_dir + '/' + left_side[i] + '.pt').numpy()

            return right_side_sum - left_side_sum
    except FileNotFoundError:
        return None

# returns 4 fingerprints - different order of reactants
def createReactionFingerprints(reaction, reactions_dir, fingerprints_dir, fp_length=8):
    try: 
        reaction_file = open(reactions_dir + "/" + reaction, 'r')
        
        reaction_split = reaction_file.read().split('-')
        left_side = reaction_split[0].split(',')
        right_side = reaction_split[1].split(',')

        compound_1 = torch.load(fingerprints_dir + '/' + left_side[0] + '.pt').numpy()
        compound_2 = torch.load(fingerprints_dir + '/' + left_side[1] + '.pt').numpy()
        compound_3 = torch.load(fingerprints_dir + '/' + right_side[0] + '.pt').numpy()
        compound_4 = torch.load(fingerprints_dir + '/' + right_side[1] + '.pt').numpy()
        
        fp_1 = np.concatenate((np.array([]), compound_1, compound_2, compound_3, compound_4))
        fp_2 = np.concatenate((np.array([]), compound_2, compound_1, compound_3, compound_4))
        fp_3 = np.concatenate((np.array([]), compound_1, compound_2, compound_4, compound_3))
        fp_4 = np.concatenate((np.array([]), compound_2, compound_1, compound_4, compound_3))

        return [fp_1, fp_2, fp_3, fp_4]

    except:
        return None

# returns single fingerprint
def createReactionFingerprint(reaction, reactions_dir, fingerprints_dir, fp_length=8):
    try: 
        reaction_file = open(reactions_dir + "/" + reaction, 'r')
        
        reaction_split = reaction_file.read().split('-')
        left_side = reaction_split[0].split(',')
        right_side = reaction_split[1].split(',')

        fp_combined = np.array([])

        for compound in left_side:
            compound_tensor = torch.load(fingerprints_dir + '/' + compound + '.pt').numpy()
            fp_combined = np.concatenate((fp_combined, compound_tensor))
            #fp_combined = torch.cat((fp_combined, compound_tensor))
        for compound in right_side:
            compound_tensor = torch.load(fingerprints_dir + '/' + compound + '.pt').numpy()
            fp_combined = np.concatenate((fp_combined, compound_tensor))
           # fp_combined = torch.cat((fp_combined, compound_tensor))
        
        return fp_combined

    except:
        return None


class ReactionsDataset(Dataset):
    def __init__(self, file_name, accepted_reaction_types) -> None:
        file = pd.read_csv(file_name)
        reactions = file['Reaction']
        types = file['Type']

        reaction_fps = []
        labels = []
        
        for reaction, type in zip(reactions, types):
            if (type in accepted_reaction_types):
                #reaction_fp = createReactionFingerprints(reaction, reactions_dir, fingerprints_dir)
                reaction_fp = createCombinedFingerprint(reaction, reactions_dir, fingerprints_dir, combine_method="average")
                if (reaction_fp is not None):
                #for fp in reaction_fp:    
                    #    reaction_fps.append(fp)
                    #    labels.append(type)
                    reaction_fps.append(reaction_fp)
                    labels.append(accepted_reaction_types.index(type)) # 0 or 1
                #reaction_fps = torch.cat(reaction_fps, reaction_fp)

        # Feature Scaling
        sc = StandardScaler()
        reaction_fps = sc.fit_transform(reaction_fps)
        
        # Convert to tensors
        fps_tensor = torch.tensor(reaction_fps, dtype=torch.float32)
        labels_tensor = torch.tensor(labels)

        self.X_train = fps_tensor
        self.y_train = labels_tensor

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(fp_length * fp_multiplying_factor, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        #self.batchnorm1 = nn.BatchNorm1d(128)
        #self.batchnorm2 = nn.BatchNorm1d(128)
    
    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        #x = self.batchnorm1(x)
        x = self.relu(self.layer2(x))
        #x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer3(x)

        return x

# Binary classification accuracy
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

# Train and evaluate model for each pair of reactions
pairs_results = {}
for pair in reaction_pairs_compare: #reaction_pairs_all:
    print("Training for pair: " + reaction_types[pair[0]] + "-" + reaction_types[pair[1]])
    # results for this pair
    pairs_results[pair] = {
        "train_accs": [], 
        "test_accs": [], 
        "types_accs": {},
        "predicted_labels": [], 
        "actual_labels": [], 
        "predicted_probabilities": [],
    }

    dataset = ReactionsDataset(reactions_csv, pair)

    # split dataset
    validation_split = .2
    dataset_size = len(dataset)
    if (dataset_size < 5):
        print("Not enough data for pair: " + reaction_types[pair[0]] + "-" + reaction_types[pair[1]])
        print("Skipping...")
        continue
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    model = ClassificationNetwork()
    model.apply(reset_weights)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for e in range(epochs):
        train_loss = 0
        train_acc = 0

        for X_batch, y_batch in trainloader:
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            
            loss = criterion(y_pred, y_batch.unsqueeze(1).type_as(y_pred))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
        
        else:
            test_loss = 0
            test_acc = 0

            with torch.no_grad():
                model.eval()
                predicted_labels_temp, actual_labels_temp, predicted_probabilities_temp = [], [], []
                
                for X_batch, y_batch in validloader:
                    y_pred = model(X_batch)
                    
                    for pred, true, probability in zip(torch.round(torch.sigmoid(y_pred)), y_batch, torch.sigmoid(y_pred)):
                        
                        #For confusion matrix
                        predicted_labels_temp.append(pred.item())
                        actual_labels_temp.append(true.item())
                        predicted_probabilities_temp.append(probability.item())
            
                    loss = criterion(y_pred, y_batch.unsqueeze(1).type_as(y_pred))
                    acc = binary_acc(y_pred, y_batch.unsqueeze(1))
                    
                    test_loss += loss.item()
                    test_acc += acc.item()
            
            model.train()

            pairs_results[pair]["train_accs"].append(train_acc / len(trainloader))
            pairs_results[pair]["test_accs"].append(test_acc / len(validloader))

            if (e == epochs - 1):
                pairs_results[pair]["predicted_labels"] = predicted_labels_temp
                pairs_results[pair]["actual_labels"] = actual_labels_temp
                pairs_results[pair]["predicted_probabilities"] = predicted_probabilities_temp

            print(f'Epoch {e+0:03}: | Train Loss: {train_loss/len(trainloader):.5f} | Train Acc: {train_acc/len(trainloader):.3f}| Test Loss: {test_loss/len(validloader):.5f} | Test Acc: {test_acc/len(validloader):.3f}')

# Compute AUC for each pair
roc_auc = []
pair_labels = []
for pair in pairs_results.keys():
    fpr, tpr, _ = roc_curve(pairs_results[pair]["actual_labels"], pairs_results[pair]["predicted_probabilities"])
    roc_auc.append(auc(fpr, tpr))
    pair_labels.append(reaction_types[pair[0]] + ' &\n' + reaction_types[pair[1]] + '\n(' + str(round(roc_auc[-1], 3)) + ')') 
plt.bar(np.arange(len(pair_labels)), roc_auc, align='center', width=0.3)
plt.xticks(np.arange(len(pair_labels)), pair_labels)
plt.ylabel("AUC")
plt.title("Pairwise classification")
plt.ylim(0, 1)
plt.show()

""" 
# Visualization
fig, axs = plt.subplots(2)
axs[0].plot(train_losses, label='Training loss')
axs[0].plot(test_losses, label='Validation loss')
axs[0].legend(frameon = False)
axs[0].set_title("Loss")
axs[1].plot(test_accs, label='Validation accuracy')
axs[1].set_title("Accuracy")
axs[1].legend(frameon = False)
axs[0].set_ylim([0, 1])
axs[1].set_ylim([0, 1])
plt.show() """