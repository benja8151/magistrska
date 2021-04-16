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

reactions_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions'
fingerprints_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/generated'
reactions_csv = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/encyme_reaction_classification/data/reactions.csv'

fp_length = 8

epochs = 20
batch_size = 150
learning_rate = 0.001

def createCombinedFingerprint(reaction, reactions_dir, fingerprints_dir, fp_length=8):
    try: 
        reaction_file = open(reactions_dir + "/" + reaction, 'r')
        
        reaction_split = reaction_file.read().split('-')
        left_side = reaction_split[0].split(',')
        right_side = reaction_split[1].split(',')

        fp_combined = torch.load(fingerprints_dir + '/' + right_side[0] + '.pt').numpy()
        fp_combined += torch.load(fingerprints_dir + '/' + right_side[1] + '.pt').numpy()
        fp_combined -= torch.load(fingerprints_dir + '/' + left_side[0] + '.pt').numpy()
        fp_combined -= torch.load(fingerprints_dir + '/' + left_side[1] + '.pt').numpy()
        
        return fp_combined

    except:
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
    def __init__(self, file_name) -> None:
        file = pd.read_csv(file_name)
        reactions = file['Reaction']
        hasRBP = file['hasRBP']
        reactionType = file['reactionType']

        reaction_fps = []
        labels = []
        types = []

        reaction_count = 0
        count = {
            0: 0,
            1:0,
        }
        
        for reaction, label, reactionType in zip(reactions, hasRBP, reactionType):
            reaction_fp = createReactionFingerprints(reaction, reactions_dir, fingerprints_dir)
            #reaction_fp = createCombinedFingerprint(reaction, reactions_dir, fingerprints_dir)
            if (reaction_fp is not None):
                for fp in reaction_fp:    
                    reaction_fps.append(fp)
                    labels.append(label)
                    types.append(reactionType)
                    reaction_count += 1
                    count[label] += 1
                """ reaction_fps.append(reaction_fp)
                labels.append(label)
                reaction_count += 1
                count[label] += 1 """
               #reaction_fps = torch.cat(reaction_fps, reaction_fp)
        
        for x in count:
            count[x] = 100 * count[x] / reaction_count

        print(reaction_count)
        print(count)
        
        # Feature Scaling
        sc = StandardScaler()
        reaction_fps = sc.fit_transform(reaction_fps)
        
        # Convert to tensors
        fps_tensor = torch.tensor(reaction_fps, dtype=torch.float32)
        labels_tensor = torch.tensor(labels)
        types_tensor = torch.tensor(types)

        self.X_train = fps_tensor
        self.y_train = labels_tensor
        self.z_train = types_tensor

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index], self.z_train[index]

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(fp_length * 4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
    
    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer2(x))
        x = self.batchnorm2(x)
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

dataset = ReactionsDataset('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/encyme_reaction_classification/data/reactions.csv')

# split dataset
validation_split = .2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

model = ClassificationNetwork()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()

train_accs, test_accs, types_accs = [], [], {}
labels_final_count = {}

for e in range(1, epochs+1):
    train_loss = 0
    train_acc = 0
    for X_batch, y_batch, _ in trainloader:
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

        accuracies_detailed = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
        }

        labels_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
        } 

        with torch.no_grad():
            model.eval()
            for X_batch, y_batch, z_batch in validloader:
                y_pred = model(X_batch)

                for pred, true, label in zip(torch.round(torch.sigmoid(y_pred)), y_batch, z_batch):
                    labels_count[label.item()] += 1
                    if (pred.item() == true.item()):
                        accuracies_detailed[label.item()] += 1
        
                loss = criterion(y_pred, y_batch.unsqueeze(1).type_as(y_pred))
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))
                
                test_loss += loss.item()
                test_acc += acc.item()
        
        for label in accuracies_detailed:
            try:
                accuracies_detailed[label] = accuracies_detailed[label] / labels_count[label]
            except: #division by 0
                continue
        
        types_accs = accuracies_detailed
        labels_final_count = labels_count

        model.train()

        train_accs.append(train_acc/len(trainloader))
        test_accs.append(test_acc/len(validloader))

    print(f'Epoch {e+0:03}: | Train Loss: {train_loss/len(trainloader):.5f} | Train Acc: {train_acc/len(trainloader):.3f}| Test Loss: {test_loss/len(validloader):.5f} | Test Acc: {test_acc/len(validloader):.3f}')

# Plot accuracies
fig, axs = plt.subplots(2)
axs[0].plot(train_accs, label='Training accuracy')
axs[0].plot(test_accs, label='Validation accuracy')
axs[0].legend(frameon=False)
axs[0].set_title("RBP detection")
axs[0].set_ylim([0, 100])

axs[1].set_title("Accuracy by Type")
print(list(types_accs.values()))
axs[1].bar(np.arange(8), list(types_accs.values()))
axs[1].set_xticks(np.arange(8))
axs[1].set_xticklabels((
    f"Hydrolase\n({labels_final_count[0]})", 
    f"Isomerase\n({labels_final_count[1]})", 
    f"Ligase\n({labels_final_count[2]})", 
    f"Lyase\n({labels_final_count[3]})", 
    f"Oxidoreductase\n({labels_final_count[4]})", 
    f"Transferase\n({labels_final_count[5]})", 
    f"Translocase\n({labels_final_count[6]})", 
    f"Unassigned\n({labels_final_count[7]})"
))

plt.show()