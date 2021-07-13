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
from sklearn.metrics import roc_curve, auc
import copy
import time

import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F



batch_size = 20
device = 'cpu'

# Dataset
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
        types = file['hasRBP']

        print("Loading graphs...")
        count = 0
        for reaction, type in zip(reactions, types):
            
            # TODO: temp
            #if (count > 5):
                #break
            
            g = self.loadGraph(self.raw_dir + '/' + reaction)
            if (g is not None):
                self.graphs.append(g)
                self.labels.append(type)
            
            print(str(round((count/len(reactions) * 100), 2)) + "%")
            count +=1

        self.labels = torch.LongTensor(self.labels)

    def loadGraph(self, path: str):
        try:
            with open(path, 'rb') as file:
                g = pickle.load(file)
            file.close()
            return g
        except:
            return None

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

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
        return os.path.exists(os.path.join(self.save_path, 'dataset.bin'))

# Helpers
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        full_atom_feature_dims = [54, 8, 12, 12, 10, 6, 2]

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

        full_bond_feature_dims = [7]

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
    model.train()
    
    train_loss = []
    for g, labels in data_loader:
        g = g.to(device)
        labels = labels.to(torch.float32).to(device)

        logits = model(g, g.edata['feat'], g.ndata['feat'])


        loss = loss_fn(logits.flatten(), labels)
        #print(loss)
        train_loss.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    return sum(train_loss) / len(train_loss)

@torch.no_grad()
def test(model, device, data_loader):
    model.eval()
    y_true, y_pred = [], []

    for g, labels in data_loader:
        g = g.to(device)
        logits = model(g, g.edata['feat'], g.ndata['feat'])
        y_true.append(labels.detach().cpu())
        y_pred.append(logits.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

# Collate function for ordinary graph classification 
def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    if isinstance(labels[0], torch.Tensor):
        return batched_graph, torch.stack(labels)
    else:
        return batched_graph, labels

#########################################################################

dataset = ReactionsDataset(
    '/home/bsmrdelj/local/git/magistrska/dgl/data/graphs_homogenous',
    '/home/bsmrdelj/local/git/magistrska/dgl/data/dataset_RBP',
    '/home/bsmrdelj/local/git/magistrska/classification/encyme_reaction_classification/data/reactions_all.csv',
    force_reload=False
)

num_reactions = len(dataset)
num_train = int(num_reactions * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_reactions))

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
out_dim = 1
epochs = 1000

#g = dataset[0][0]
#print(g.ndata['feats'])
#print(g.ntypes)
#print(g.etypes)
#print(num_reactions)

# load model
model = DeeperGCN(
    node_feat_dim=node_feat_dim,
    edge_feat_dim=edge_feat_dim,
    hid_dim=256,
    out_dim=out_dim,
    num_layers=7,
    dropout=0.2,
    learn_beta=True
).to(device)
print(model)

opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

 # training & validation & testing
best_auc = 0
best_model = copy.deepcopy(model)
times = []

print('---------- Training ----------')
for i in range(1000):
    t1 = time.time()
    train_loss = train(model, device, train_dataloader, opt, loss_fn)
    t2 = time.time()

    if i >= 5:
        times.append(t2 - t1)

    train_auc = test(model, device, train_dataloader)
    valid_auc = test(model, device, test_dataloader,)

    print(f'Epoch {i} | Train Loss: {train_loss:.4f} | Train Auc: {train_auc:.4f} | Valid Auc: {valid_auc:.4f}')

    if valid_auc > best_auc:
        best_auc = valid_auc
        best_model = copy.deepcopy(model)

print('---------- Testing ----------')
test_auc = test(best_model, device, test_dataloader,)
print(f'Test Auc: {test_auc}')
if len(times) > 0:
    print('Times/epoch: ', sum(times) / len(times))