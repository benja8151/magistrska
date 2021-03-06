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


reactions_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/ParsedAll'
fingerprints_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/generated'
reactions_csv = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/reaction_type_classification/data/reactions.csv'

n_types = 8
fp_length = 8
epochs = 50
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
    def __init__(self, file_name) -> None:
        file = pd.read_csv(file_name)
        reactions = file['Reaction']
        types = file['Type']

        reaction_fps = []
        labels = []
        
        for reaction, type in zip(reactions, types):
            reaction_fp = createReactionFingerprints(reaction, reactions_dir, fingerprints_dir)
            #reaction_fp = createCombinedFingerprint(reaction, reactions_dir, fingerprints_dir)
            if (reaction_fp is not None):
                for fp in reaction_fp:    
                    reaction_fps.append(fp)
                    labels.append(type)

                """ reaction_fps.append(reaction_fp)
                labels.append(type) """
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
        self.layer1 = nn.Linear(fp_length * 4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_types)

        self.relu = F.relu
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        x = self.dropout(self.relu(self.layer1(inputs)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))
        x = F.log_softmax(self.layer4(x), dim=1)

        return x

dataset = ReactionsDataset(reactions_csv)

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
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

train_losses, test_losses, test_accs, types_accs = [], [], [], {}
labels_final_count = {}
for e in range(epochs):
    running_loss = 0
    for fps, labels in trainloader:
        optimizer.zero_grad()
        log_ps = model(fps)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    else:
        test_loss = 0
        accuracy = 0

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
            for fps, labels in validloader:
                log_ps = model(fps)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                for label, val in zip(labels, equals):
                    if (val):
                        accuracies_detailed[label.item()]+=1
                    labels_count[label.item()]+=1

                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        for label in accuracies_detailed:
            try:
                accuracies_detailed[label] = accuracies_detailed[label] / labels_count[label]
            except: #division by 0
                continue
        
        types_accs = accuracies_detailed
        labels_final_count = labels_count

        model.train()

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(validloader))
        test_accs.append(accuracy/len(validloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

fig, axs = plt.subplots(3)
axs[0].plot(train_losses, label='Training loss')
axs[0].plot(test_losses, label='Validation loss')
axs[0].legend(frameon = False)
axs[0].set_title("Loss")
axs[1].plot(test_accs, label='Validation accuracy')
axs[1].set_title("Accuracy")
axs[1].legend(frameon = False)

axs[2].set_title("Accuracy by Type")
print(list(types_accs.values()))
axs[2].bar(np.arange(8), list(types_accs.values()))
axs[2].set_xticks(np.arange(8))
axs[2].set_xticklabels((
    f"Hydrolase\n({labels_final_count[0]})", 
    f"Isomerase\n({labels_final_count[1]})", 
    f"Ligase\n({labels_final_count[2]})", 
    f"Lyase\n({labels_final_count[3]})", 
    f"Oxidoreductase\n({labels_final_count[4]})", 
    f"Transferase\n({labels_final_count[5]})", 
    f"Translocase\n({labels_final_count[6]})", 
    f"Unassigned\n({labels_final_count[7]})"
))

axs[0].set_ylim([0, 1])
axs[1].set_ylim([0, 1])
plt.show()