################################
# USED FOR RUNNING ON REMOTE GPU
################################

import dgl
import dgl.function as dglfn
from dgl.data import DGLDataset
import os
from dgl.data.graph_serialize import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.functional import edge_softmax
from dgl.readout import mean_nodes
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
import copy
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

torch.cuda.empty_cache()
final_results_dir = '/home/bsmrdelj/local/git/magistrska/dgl/classification/results/reaction_type'

epochs = 1000
n_types = 8
batch_size = 200
learning_rate = 0.001
k_folds = 5
device = 'cuda:0'

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
type_colors = [
    "#1700C5",
    "#E2A2FF",
    "#3FBEA8",
    "#24849D",
    "#4D4351",
    "#481165",
    "#FF9E73",
    "#CE6942",
]


# Dataset
class ValidationDataset(DGLDataset):
    def __init__(self, X, y) -> None:
        self.X_valid = X
        self.y_valid = y
        super(ValidationDataset, self).__init__(
            name='ValidationDataset',
            raw_dir='/home/bsmrdelj/local/git/magistrska/dgl/data/graphs_with_master_node',
            save_dir='/home/bsmrdelj/local/git/magistrska/dgl/data/dataset_reaction_type',
            verbose=True,
            force_reload=True
        )

        def __len__(self):
            return len(self.y_valid)

        def __getitem__(self, index):
            return self.X_valid[index], self.y_valid[index]

class ReactionsDataset(DGLDataset):
    def __init__(self, raw_dir: str, save_dir: str, labelsFile: str, force_reload = False):
        self.labelsFile = labelsFile
        super(ReactionsDataset, self).__init__(
            name='ReactionsDataset',
            raw_dir=raw_dir,
            save_dir=save_dir,
            verbose=True,
            force_reload=force_reload
        )

    def process(self):
        self.graphs = []
        self.labels = []

        file = pd.read_csv(self.labelsFile)
        reactions = file['Reaction']
        types = file['Type']

        print("Loading graphs...")
        count = 0
        for reaction, type in zip(reactions, types):
            
            #TODO: temp
            #if (count > 5):
                #break
            
            g = self.loadGraph(self.raw_dir + '/' + reaction)
            if (g is not None):
                self.graphs.append(g)
                self.labels.append(type)
            
            print(str(round((count/len(reactions) * 100), 2)) + "%")
            count +=1
        
        self.graphs = torch.tensor(self.graphs)
        self.labels = torch.LongTensor(self.labels)

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.1).split(self.graphs, self.labels)
        train_index, valid_index = next(split)

        # Training data
        self.X_train = self.graphs[train_index]
        self.y_train = self.labels[train_index]

        # Final validation data
        self.X_valid = self.graphs[valid_index]
        self.y_valid = self.labels[valid_index]

    def generatevalidationDataset(self):
        return ValidationDataset(self.X_valid, self.y_valid)

    def loadGraph(self, path: str):
        try:
            with open(path, 'rb') as file:
                g = pickle.load(file)
            file.close()
            return g
        except:
            return None

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def __len__(self):
        return len(self.y_train)

    def save(self):
        print("Saving dataset...")
        save_graphs(os.path.join(self.save_path, 'dataset.bin'), self.graphs, {'labels': self.labels})

    def load(self):
        print("Loading dataset from cache...")
        graphs, labels = load_graphs(os.path.join(self.save_path, 'dataset.bin'))
        self.graphs = graphs
        self.labels = labels['labels']
    
    def has_cache(self):
        print("Checking if cached file exists...")
        return os.path.exists(os.path.join(self.save_dir, 'dataset.bin'))

# Helpers
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        full_atom_feature_dims = [81, 8, 12, 12, 10, 6, 2]

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            #torch.set_printoptions(threshold=10_000)
            #print(x[:,i].long())
            try:
                x_embedding += self.atom_embedding_list[i](x[:,i].long())
            except:
                torch.set_printoptions(threshold=10_000)
                print(x[:,i].long())
                raise Exception("Woops")

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        full_bond_feature_dims = [9]

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i].long())

        return bond_embedding

# Neural Network
class MLP(nn.Sequential):
    r"""

    Description
    -----------
    From equation (5) in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    """
    def __init__(self,
                 channels,
                 act='relu',
                 dropout=0.,
                 bias=True):
        layers = []
        
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm1d(channels[i], affine=True))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        super(MLP, self).__init__(*layers)

class MessageNorm(nn.Module):
    r"""
    
    Description
    -----------
    Message normalization was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"

    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """
    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=learn_scale)

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale

class GENConv(nn.Module):
    r"""
    
    Description
    -----------
    Generalized Message Aggregator was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"

    Parameters
    ----------
    in_dim: int
        Input size.
    out_dim: int
        Output size.
    aggregator: str
        Type of aggregation. Default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 aggregator='softmax',
                 beta=1.0,
                 learn_beta=False,
                 p=1.0,
                 learn_p=False,
                 msg_norm=False,
                 learn_msg_scale=False,
                 mlp_layers=1,
                 eps=1e-7):
        super(GENConv, self).__init__()
        
        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True) if learn_beta and self.aggr == 'softmax' else beta
        self.p = nn.Parameter(torch.Tensor([p]), requires_grad=True) if learn_p else p

        self.edge_encoder = BondEncoder(in_dim)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # Node and edge feature size need to match.
            g.ndata['h'] = node_feats
            g.edata['h'] = self.edge_encoder(edge_feats)
            g.apply_edges(dglfn.u_add_e('h', 'h', 'm'))

            if self.aggr == 'softmax':
                g.edata['m'] = F.relu(g.edata['m']) + self.eps
                g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             dglfn.sum('x', 'm'))
            
            elif self.aggr == 'power':
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.edata['m'], minv, maxv)
                g.update_all(lambda edge: {'x': torch.pow(edge.data['m'], self.p)},
                             dglfn.mean('x', 'm'))
                torch.clamp_(g.ndata['m'], minv, maxv)
                g.ndata['m'] = torch.pow(g.ndata['m'], self.p)
            
            else:
                raise NotImplementedError(f'Aggregator {self.aggr} is not supported.')
            
            if self.msg_norm is not None:
                g.ndata['m'] = self.msg_norm(node_feats, g.ndata['m'])
            
            feats = node_feats + g.ndata['m']
            
            return self.mlp(feats)

class DeeperGCN(nn.Module):
    r"""

    Description
    -----------
    Introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"

    Parameters
    ----------
    node_feat_dim: int
        Size of node feature.
    edge_feat_dim: int
        Size of edge feature.
    hid_dim: int
        Size of hidden representations.
    out_dim: int
        Size of output.
    num_layers: int
        Number of graph convolutional layers.
    dropout: float
        Dropout rate. Default is 0.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable weight. Default is False.
    aggr: str
        Type of aggregation. Default is 'softmax'.
    mlp_layers: int
        Number of MLP layers in message normalization. Default is 1.
    """
    def __init__(self,
                 node_feat_dim,
                 edge_feat_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout=0.,
                 beta=1.0,
                 learn_beta=False,
                 aggr='softmax',
                 mlp_layers=1):
        super(DeeperGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            conv = GENConv(in_dim=hid_dim,
                           out_dim=hid_dim,
                           aggregator=aggr,
                           beta=beta,
                           learn_beta=learn_beta,
                           mlp_layers=mlp_layers)
            
            self.gcns.append(conv)
            self.norms.append(nn.BatchNorm1d(hid_dim, affine=True))

        self.node_encoder = AtomEncoder(hid_dim)
        self.pooling = AvgPooling()
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, edge_feats, node_feats=None):
        with g.local_scope():
            hv = self.node_encoder(node_feats)
            he = edge_feats

            for layer in range(self.num_layers):
                hv1 = self.norms[layer](hv)
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                hv = self.gcns[layer](g, hv1, he) + hv

            h_g = self.pooling(g, hv)

            return self.output(h_g)

# Train/Test functions

def train(model, device, data_loader, opt, loss_fn):
    """
    Returns train loss, predicted labels and actual labels
    """

    model.train()

    train_loss = 0
    predicted_labels_train = []
    actual_labels_train = []
    
    for g, labels in data_loader:
        opt.zero_grad()
        g = g.to(device)
        labels = labels.to(device)

        log_ps = model(g, g.edata['feat'], g.ndata['feat'])
        _, top_class = torch.exp(log_ps).topk(1, dim=1)
        
        actual_labels_batch = labels.cpu().numpy().astype(int)
        predicted_labels_batch = top_class.detach().numpy().flatten()

        predicted_labels_train = np.concatenate((predicted_labels_train, predicted_labels_batch))
        actual_labels_train = np.concatenate((actual_labels_train, actual_labels_batch))

        loss = loss_fn(log_ps, labels)
        train_loss.append(loss.item())
        
        loss.backward()
        opt.step()
        train_loss += loss.item()

    return train_loss / len(data_loader), predicted_labels_train, actual_labels_train

@torch.no_grad()
def test(model, device, data_loader, loss_fn):
    """
    Returns test loss, predicted labels and actual labels
    """
    model.eval()

    predicted_labels_test = []
    actual_labels_test = []

    test_loss = 0

    for g, labels in data_loader:
        g = g.to(device)
        log_ps = model(g, g.edata['feat'], g.ndata['feat'])
        test_loss += loss_fn(log_ps, labels).item()
        ps = torch.exp(log_ps)
        _, top_class = ps.topk(1, dim=1)

        actual_labels_batch = labels.cpu().numpy().astype(int)
        predicted_labels_batch = top_class.detach().numpy().flatten()

        predicted_labels_test = np.concatenate((predicted_labels_test, predicted_labels_batch))
        actual_labels_test = np.concatenate((actual_labels_test, actual_labels_batch))

    return test_loss / len(data_loader), predicted_labels_test, actual_labels_test

# Collate function for ordinary graph classification 
def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    if isinstance(labels[0], torch.Tensor):
        return batched_graph, torch.stack(labels)
    else:
        return batched_graph, labels

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def classification_metrics(actual_labels, predicted_labels):
    cm = confusion_matrix(actual_labels, predicted_labels)
    
    FP = cm.sum(axis=0) - np.diag(cm) 
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP) 

    # Class-wise calculations
    sensitivity = np.nan_to_num(TP/(TP+FN))
    specificity = np.nan_to_num(TN/(TN+FP))
    precision = np.nan_to_num(TP/(TP+FP))
    f1 = np.nan_to_num(2 * (precision * sensitivity) / (precision + sensitivity))

    # Macro calculations
    macro_accuracy = 100 * np.mean(np.sum(TP) / len(actual_labels))
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    macro_precision = np.mean(precision)
    macro_f1 = np.mean(f1)

    # Weighted calculations
    unique, counts = np.unique(actual_labels, return_counts=True)
    if len(unique) < n_types:
        print("##################ERROR#################")
    countDict = dict(zip(unique, counts))
    weights = [countDict[type]/len(actual_labels) for type in unique]
    weighted_sensitivity = np.average(sensitivity, weights=weights)
    weighted_specificity = np.average(specificity, weights=weights)
    weighted_precision = np.average(precision, weights=weights)
    weighted_f1 = np.average(f1, weights=weights)

    return {
        "weighted": {
            "sensitivity": weighted_sensitivity,
            "specificity": weighted_specificity,
            "precision": weighted_precision,
            "f1": weighted_f1
        },
        "macro": {
            "accuracy": macro_accuracy, 
            "sensitivity": macro_sensitivity, 
            "specificity": macro_specificity, 
            "precision": macro_precision,
            "f1": macro_f1
        },
        "classwise": {
            "sensitivity": sensitivity, 
            "specificity": specificity, 
            "precision": precision,
            "f1": f1
        }
    }

#########################################################################

dataset = ReactionsDataset(
    '/home/bsmrdelj/local/git/magistrska/dgl/data/graphs_with_master_node',
    '/home/bsmrdelj/local/git/magistrska/dgl/data/dataset_reaction_type',
    '/home/bsmrdelj/local/git/magistrska/classification/reaction_type_classification/data/reactions_all.csv',
    force_reload=True
)
validation_dataset = dataset.generatevalidationDataset()

# k-fold cross validation
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
k_fold_results = {}

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset.graphs, dataset.labels)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # results for this fold
    k_fold_results[fold] = {
        "train": {
            "weighted": {
                "sensitivity": [], 
                "specificity": [], 
                "precision": [],
                "f1": []
            },
            "macro": {
                "accuracy": [],
                "sensitivity": [], 
                "specificity": [], 
                "precision": [],
                "f1": []
            },
            "classwise": {
                "sensitivity": [], 
                "specificity": [], 
                "precision": [],
                "f1": []
            },
            "loss": [],
        }, 
        "test": {
            "weighted": {
                "sensitivity": [], 
                "specificity": [], 
                "precision": [],
                "f1": []
            },
            "macro": {
                "accuracy": [],
                "sensitivity": [], 
                "specificity": [], 
                "precision": [],
                "f1": []
            },
            "classwise": {
                "sensitivity": [], 
                "specificity": [], 
                "precision": [],
                "f1": []
            },
            "loss": [],
        },
        "valid": {
            "weighted": {
                "sensitivity": 0, 
                "specificity": 0, 
                "precision": 0,
                "f1": 0
            },
            "macro": {
                "accuracy": 0,
                "sensitivity": 0, 
                "specificity": 0, 
                "precision": 0,
                "f1": 0
            },
            "classwise": {
                "sensitivity": 0, 
                "specificity": 0, 
                "precision": 0,
                "f1": 0
            },
            "loss": 0,
        },

        # For Classification Matrix and Mosaic:
        "predicted_labels_all": [],
        "actual_labels_all": [],
        "predicted_probabilities_all": [],

        # OneVsRest results (for ROC curve): 
        "predicted_labels": [[] for i in range(n_types)], 
        "actual_labels": [[] for i in range(n_types)], 
        "predicted_probabilities": [[] for i in range(n_types)]
    }
    
    # sampling and loading data
    train_sampler = SubsetRandomSampler(train_ids)
    test_sampler = SubsetRandomSampler(test_ids)

    train_dataloader = GraphDataLoader(
        dataset,
        sampler = train_sampler,
        batch_size=batch_size,
        drop_last = False,
        collate_fn=collate_dgl
    )
    test_dataloader = GraphDataLoader(
        dataset,
        sampler = test_sampler,
        batch_size=batch_size,
        drop_last = False,
        collate_fn=collate_dgl
    )

    node_feat_dim = dataset[0][0].ndata['feat'].size()[-1]
    edge_feat_dim = dataset[0][0].edata['feat'].size()[-1]

    # load model
    model = DeeperGCN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hid_dim=256,
        out_dim=n_types,
        num_layers=7,
        dropout=0.2,
        learn_beta=True
    ).to(device)
    model.apply(reset_weights)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # For final check
    best_f1_score = 0
    best_model = None

    for e in range(epochs):
        print('---------- Training ----------')    
        train_loss, predicted_labels_train, actual_labels_train = train(model, device, train_dataloader, opt, loss_fn)
        print('---------- Testing ----------')
        test_loss, predicted_labels_test, actual_labels_test = test(model, device, test_dataloader, loss_fn)    

        k_fold_results[fold]["train"]["loss"].append(train_loss)
        k_fold_results[fold]["test"]["loss"].append(test_loss)

        classification_results_train = classification_metrics(actual_labels_train, predicted_labels_train)
        classification_results_test = classification_metrics(actual_labels_test, predicted_labels_test)

        for (type, results) in classification_results_train.items():
            for (metric, value) in results.items():
                k_fold_results[fold]["train"][type][metric].append(value)
        for (type, results) in classification_results_test.items():
            for (metric, value) in results.items():
                k_fold_results[fold]["test"][type][metric].append(value)

        print(
            "Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(k_fold_results[fold]["train"]["loss"][-1]),
            "Test Loss: {:.3f}.. ".format(k_fold_results[fold]["test"]["loss"][-1]),
            "Test Accuracy: {:.3f}".format(k_fold_results[fold]["test"]["macro"]["accuracy"][-1])
        )

         # Save best model for later
        if (best_f1_score < classification_results_test["weighted"]["f1"]):
            best_f1_score = classification_results_test["weighted"]["f1"]
            print("New best model found with F1 score of: ", best_f1_score)
            best_model = copy.deepcopy(model.state_dict())
    
    # At the end of each fold, make predictions on validation dataset, all in one batch
    model.apply(reset_weights)
    model.load_state_dict(best_model)
    # Save final model
    torch.save(model.state_dict(), f"{final_results_dir}/models/model_{fold}.pt")
    
    model.eval()
    with torch.no_grad():
        
        g = validation_dataset.X_valid.to(device)
        labels = validation_dataset.y_valid.to(device)

        log_ps = model(g, g.edata['feat'], g.ndata['feat'])
        valid_loss = loss_fn(log_ps, labels).item()
        
        ps = torch.exp(log_ps)
        _, top_class = ps.topk(1, dim=1)

        actual_labels_valid = labels.cpu().numpy().astype(int)
        predicted_labels_valid = top_class.detach().numpy().flatten()
        
        k_fold_results[fold]["predicted_labels_all"] = predicted_labels_valid
        k_fold_results[fold]["actual_labels_all"] = actual_labels_valid
        k_fold_results[fold]["predicted_probabilities_all"] = ps
        k_fold_results[fold]["valid"]["loss"] = valid_loss
        
        classification_results_valid = classification_metrics(actual_labels_valid, predicted_labels_valid)
        for (type, results) in classification_results_valid.items():
            for (metric, value) in results.items():
                k_fold_results[fold]["valid"][type][metric] = value

        print(
            f'Validation results for fold {fold}: ',
            "Loss: {:.3f}.. ".format(valid_loss),
            "Accuracy: {:.3f}".format(k_fold_results[fold]["valid"]["macro"]["accuracy"]),
            "F1: {:.3f}".format(k_fold_results[fold]["valid"]["weighted"]["f1"])
        )

    model.train()

print(k_fold_results)