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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from confusion_matrix import make_confusion_matrix
from classification_mosaic import nclass_classification_mosaic_plot
from itertools import cycle
from sklearn.model_selection import StratifiedKFold

reactions_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions'
fingerprints_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/fingerprint_generator_new/generated'
reactions_csv = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/reaction_type_classification/data/reactions_all.csv'

n_types = 8
fp_length = 128
fp_multiplying_factor = 1
epochs = 50
batch_size = 150
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
    def __init__(self, file_name) -> None:
        file = pd.read_csv(file_name)
        reactions = file['Reaction']
        types = file['Type']

        reaction_fps = []
        labels = []
        
        for reaction, type in zip(reactions, types):
            #reaction_fp = createReactionFingerprints(reaction, reactions_dir, fingerprints_dir)
            reaction_fp = createCombinedFingerprint(reaction, reactions_dir, fingerprints_dir, combine_method="average")
            if (reaction_fp is not None):
               #for fp in reaction_fp:    
                #    reaction_fps.append(fp)
                #    labels.append(type)
                reaction_fps.append(reaction_fp)
                labels.append(type)
               #reaction_fps = torch.cat(reaction_fps, reaction_fp)

        # Feature Scaling
        sc = StandardScaler()
        reaction_fps = sc.fit_transform(reaction_fps)
        
        # Convert to tensors
        fps_tensor = torch.tensor(reaction_fps, dtype=torch.float32)
        labels_tensor = torch.tensor(labels)

        print(len(reaction_fps))

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

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

dataset = ReactionsDataset(reactions_csv)

# k-fold cross validation
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
k_fold_results = {}

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset.X_train, dataset.y_train)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # counting
    """  counts_train = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    counts_test = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    
    for id in test_ids:
        type = dataset.y_train[id].item()
        counts_test[type] = counts_test[type] + 1
    
    for id in train_ids:
        type = dataset.y_train[id].item()
        counts_train[type] = counts_train[type] + 1

    for key in counts_train.keys():
        counts_train[key] = round(counts_train[key] / len(dataset.y_train), 2)
    for key in counts_test.keys():
        counts_test[key] = round(counts_test[key] / len(dataset.y_train), 2)

    print(counts_train)
    print(counts_test) """

    # results for this fold
    k_fold_results[fold] = {
        "train_losses": [], 
        "test_losses": [], 
        "test_accs": [],
        # OneVsRest results (for ROC curve): 
        "predicted_labels": [[] for i in range(n_types)], 
        "actual_labels": [[] for i in range(n_types)], 
        "predicted_probabilities": [[] for i in range(n_types)]
    }

    # sampling and loading data
    train_sampler = SubsetRandomSampler(train_ids)
    test_sampler = SubsetRandomSampler(test_ids)

    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    model = ClassificationNetwork()
    model.apply(reset_weights)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)


    #train_losses, test_losses, test_accs = [], [], []
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

            with torch.no_grad():
                model.eval()
                for fps, labels in testloader:
                    log_ps = model(fps)
                    test_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)

                    
                    # For Confuxion Matrix
                    #for label, prediction in zip(labels, top_class):
                    #    actual_labels.append(label.item())
                    #    predicted_labels.append(prediction.item())

                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()

            k_fold_results[fold]["train_losses"].append(running_loss / len(trainloader))
            k_fold_results[fold]["test_losses"].append(test_loss / len(testloader))
            k_fold_results[fold]["test_accs"].append(accuracy / len(testloader))
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(k_fold_results[fold]["train_losses"][-1]),
                "Test Loss: {:.3f}.. ".format(k_fold_results[fold]["test_losses"][-1]),
                "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

            
            # Predict values for ROC curve using model trained in last epoch
            if (e == epochs-1):
                 for i in range(n_types):
                    for fps, labels in testloader:
                        log_ps = model(fps)
                        ps = torch.exp(log_ps)
                        _, top_class = ps.topk(1, dim=1)
                        for label, prediction, probabilities in zip(labels, top_class, ps):
                            k_fold_results[fold]["actual_labels"][i].append(1 if label.item() == i else 0)
                            k_fold_results[fold]["predicted_labels"][i].append(1 if prediction.item()==i else 0)
                            k_fold_results[fold]["predicted_probabilities"][i].append(probabilities[i].item())


# Average result accross all folds
average_results = {"train_losses": [], "test_losses": [], "test_accs": []}
for i in range(epochs):
    train_loss = 0
    test_loss = 0
    test_acc = 0

    for fold in k_fold_results.keys():
        train_loss += k_fold_results[fold]["train_losses"][i]
        test_loss += k_fold_results[fold]["test_losses"][i]
        test_acc += k_fold_results[fold]["test_accs"][i]

    average_results["train_losses"].append(train_loss / len(k_fold_results.keys()))
    average_results["test_losses"].append(test_loss / len(k_fold_results.keys()))
    average_results["test_accs"].append(test_acc / len(k_fold_results.keys()))


""" # Predictions over entire dataset
with torch.no_grad():
    predicted_labels, actual_labels, predicted_probabilities = [], [], []
    model.eval()
    for fps, labels in DataLoader(dataset, batch_size=batch_size):
        log_ps = model(fps)
        ps = torch.exp(log_ps)
        _, top_class = ps.topk(1, dim=1)
        for label, prediction, probabilities in zip(labels, top_class, ps):
            actual_labels.append(label.item())
            predicted_labels.append(prediction.item())
            predicted_probabilities.append(probabilities[label].item()) """

""" # Confusion Matrix
make_confusion_matrix(
    confusion_matrix(actual_labels, predicted_labels),
    categories = reaction_types,
    figsize = (8, 6),
    sum_stats = True
) """

""" # Classification Mosaic
results = [[0 for i in range(n_types)] for i in range(n_types)]
for (pred, actual) in zip(predicted_labels, actual_labels):
    results[actual][pred] += 1
nclass_classification_mosaic_plot(n_types, results, reaction_types) """

""" # OneVsRest Predictions(entire dataset)
with torch.no_grad():
    predicted_labels, actual_labels, predicted_probabilities = [[] for i in range(n_types)], [[] for i in range(n_types)], [[] for i in range(n_types)]
    model.eval()
    for i in range(n_types):
        for fps, labels in DataLoader(dataset, batch_size=batch_size):
            log_ps = model(fps)
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            for label, prediction, probabilities in zip(labels, top_class, ps):
                actual_labels[i].append(1 if label.item() == i else 0)
                predicted_labels[i].append(1 if prediction.item()==i else 0)
                predicted_probabilities[i].append(probabilities[i].item()) """


""" #Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_types):
    fpr[i], tpr[i], _ = roc_curve(actual_labels[i], predicted_probabilities[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
for i, color in zip(range(n_types), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label='{0} (area = {1:0.2f})'
             ''.format(reaction_types[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for each class')
plt.legend(loc="lower right")
plt.show() """

#Compute ROC curve and ROC area for each class for each fold
fig, axs = plt.subplots(k_folds)
for fold in range(k_folds):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_types):
        fpr[i], tpr[i], _ = roc_curve(k_fold_results[fold]["actual_labels"][i], k_fold_results[fold]["predicted_probabilities"][i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    for i, color in zip(range(n_types), colors):
        axs[fold].plot(fpr[i], tpr[i], color=color,
                label='{0} (area = {1:0.2f})'
                ''.format(reaction_types[i], roc_auc[i]))

    axs[fold].plot([0, 1], [0, 1], 'k--')
    axs[fold].set_xlim([0.0, 1.0])
    axs[fold].set_ylim([0.0, 1.05])
    axs[fold].set_xlabel('False Positive Rate')
    axs[fold].set_ylabel('True Positive Rate')
    axs[fold].set_title('Fold ' + str(fold))
    axs[fold].legend(loc="lower right")
fig.suptitle('ROC for each class')
plt.show()


# Visualization
fig, axs = plt.subplots(2)
#axs[0].plot(train_losses, label='Training loss')
#axs[0].plot(test_losses, label='Validation loss')
for key in k_fold_results:
    axs[0].plot(k_fold_results[key]["train_losses"], label="Training Loss(" + str(key) + ")")
    axs[0].plot(k_fold_results[key]["test_losses"], label="Test Loss(" + str(key) + ")")
axs[0].plot(average_results["train_losses"], label="Average Training Loss", linewidth = 4)
axs[0].plot(average_results["test_losses"], label="Average Test Loss", linewidth = 4)
axs[0].legend(frameon = False)
axs[0].set_title("Loss")

#axs[1].plot(test_accs, label='Validation accuracy')
for key in k_fold_results:
    axs[1].plot(k_fold_results[key]["test_accs"], label="Test Accuracy(" + str(key) + ")")
    axs[1].plot(k_fold_results[key]["test_accs"], label="Test Accuracy(" + str(key) + ")")
axs[1].plot(average_results["test_accs"], label="Average Test Accuracy", linewidth = 4)
axs[1].set_title("Accuracy")
axs[1].legend(frameon = False)
axs[0].set_ylim([0, 1])
axs[1].set_ylim([0, 1])
plt.show()