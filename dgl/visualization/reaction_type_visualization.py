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
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import colorsys

n_types = 8
batch_size = 1
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
        types = file['Type']

        print("Loading graphs...")
        count = 0
        for reaction, type in zip(reactions, types):
            
            #TODO: temp
            if (count > 200):
                break
            
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
        print("Forward called...")
        with g.local_scope():
            hv = self.node_encoder(node_feats)
            print("hv: ", hv.shape)
            he = edge_feats
            print("he: ", he.shape)

            for layer in range(self.num_layers):
                print("Layer: ", layer)
                hv1 = self.norms[layer](hv)
                print("Normalization: ", hv1.shape)
                hv1 = F.relu(hv1)
                print("Relu: ", hv1.shape)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                print("Dropout: ", hv1.shape)
                hv = self.gcns[layer](g, hv1, he) + hv
                print("Layer output: ", hv.shape)

            h_g = self.pooling(g, hv)
            print("Pooling: ", h_g.shape)
            
            return self.output(h_g)

# Train/Test functions
def train(model, device, data_loader, opt, loss_fn):
    model.train()
    
    train_loss = []
    for g, labels in data_loader:
        g = g.to(device)
        labels = labels.to(device)

        log_ps = model(g, g.edata['feat'], g.ndata['feat'])

        loss = loss_fn(log_ps, labels)
        train_loss.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    return sum(train_loss) / len(train_loss)

@torch.no_grad()
def test(model, device, data_loader, calculate_auc):
    model.eval()

    actual_labels, predicted_labels, predicted_probabilities = [[] for i in range(n_types)], [[] for i in range(n_types)], [[] for i in range(n_types)]

    accuracy = 0

    if (not calculate_auc):
        for g, labels in data_loader:
            g = g.to(device)
            log_ps = model(g, g.edata['feat'], g.ndata['feat'])
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.to(device).view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    else:
        #OneVsRest
        for i in range(n_types):
            for g, labels in data_loader:
                g = g.to(device)
                log_ps = model(g, g.edata['feat'], g.ndata['feat'])
                ps = torch.exp(log_ps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.to(device).view(*top_class.shape)
                
                for label, prediction, probabilities in zip(labels.detach().cpu(), top_class, ps):
                    actual_labels[i].append(1 if label.item() == i else 0)
                    predicted_labels[i].append(1 if prediction.item() == i else 0)
                    predicted_probabilities[i].append(probabilities[i].item())
                    
                # only run once
                if i == 0:
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

        for i in range(n_types):
            fpr, tpr, _  = roc_curve(actual_labels[i], predicted_probabilities[i])
            print("Type:", i, "Auc: ", auc(fpr, tpr))

    return accuracy/len(data_loader)

# Collate function for ordinary graph classification 
def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    if isinstance(labels[0], torch.Tensor):
        return batched_graph, torch.stack(labels)
    else:
        return batched_graph, labels

#########################################################################

""" 
    0: Hydrolase
    1: Isomerase
    2: Ligase
    3: Lyase
    4: Oxidoreductase
    5: Transferase
    6: Translocase
    7: Unassigned
"""

# Maps from rdkit atom indexes to DGL graph indexes  
# Currently only for graphs with master nodes  
def createAtomMappings(reactionsDir: str, reactionName: str, csvDir: str):
    reactionFile = open(reactionsDir + '/' + reactionName)
    reactionSides = reactionFile.read().split('-')
    reactionFile.close()
    leftSide = reactionSides[0].split(',')
    rightSide = reactionSides[1].split(',')

    atomIndex = 0
    atomMapping = {}

    for compound in leftSide:
        nodes = pd.read_csv(csvDir + '/' + compound + '/nodes.csv')
        atomMapping[compound] = list(range(atomIndex, len(nodes)+atomIndex))
        atomIndex += len(nodes)+1

    for compound in rightSide:
        nodes = pd.read_csv(csvDir + '/' + compound + '/nodes.csv')
        atomMapping[compound] = list(range(atomIndex, len(nodes)+atomIndex))
        atomIndex += len(nodes)+1

    for compound in leftSide + rightSide:
        atomMapping[compound] = atomMapping[compound] + [atomIndex]
        atomIndex+=1

    return atomMapping

# Runs evaluation on graph and returns class probability
def testModelOnGraph(model, testGraph, reactionType):
    ps = model(testGraph, testGraph.edata['feat'], testGraph.ndata['feat'])
    return torch.exp(ps)[reactionType]

# Returns atom color
def getColorFromDelta(delta, minDelta, maxDelta):
    if (delta < 0):
        hue=0
        saturation = 1
        brightness = delta/minDelta
    else:
        hue=0.43
        saturation = 1
        brightness = delta/maxDelta

    a=0
    b=1
    c=1
    d=0.5
    brightness = c + (((d-c)/(b-a)) * (brightness-a))

    return colorsys.hls_to_rgb(hue, brightness, saturation)

# Perfoms a series of model evaluations by removing single atoms
def evaluateModel(
    modelPath: str, 
    graphsDir: str, 
    reactionName: str, 
    reactionType: int,
    reactionsDir: str, 
    csvDir: str,
    molDir: str,
    outputDir: str
):
    
    # Load Reaction graph
    testGraph = None
    with open(graphsDir + '/' + reactionName, 'rb') as file:
        testGraph = pickle.load(file)
    file.close()
    print(testGraph)

    node_feat_dim = testGraph.ndata['feat'].size()[-1]
    edge_feat_dim = testGraph.edata['feat'].size()[-1]
    out_dim = n_types

    # Load model
    model = DeeperGCN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hid_dim=256,
        out_dim=out_dim,
        num_layers=7,
        dropout=0.2,
        learn_beta=True
    ).to(device)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    model.eval()

    # Atom mappings
    atomMappings = createAtomMappings(reactionsDir, reactionName, csvDir)

    # Calculate original graph evaluation
    originalProb = testModelOnGraph(model, testGraph, reactionType)

    # Evaluate classification when single atom is removed
    removedDeltas = {}
    maxDelta = 0
    minDelta = 0
    for compound, mapping in atomMappings:
        removedProbs = []

        # Compounds with 1 atom and 1 master node
        if len(mapping) == 2:
            removedProbs.append(modifiedGraph = dgl.remove_nodes(testGraph, mapping))
        # Larger compounds
        else:
            for i in range(len(mapping) - 1):
                modifiedGraph = dgl.remove_nodes(testGraph, mapping[i])

                removedProbs.append(testModelOnGraph(model, modifiedGraph, reactionType))

        removedDeltas[compound] = [((originalProb -  x) / originalProb) for x in removedProbs]
        maxDelta = max(maxDelta, max(removedDeltas[compound]))
        minDelta = min(minDelta, min(removedDeltas[compound]))

    # Visualization
    for compound, deltas in removedDeltas:
        
        # Load .mol file
        mol = Chem.MolFromMolFile(molDir + '/' + compound + '.mol')
        
        # Get atom colors
        colors = [getColorFromDelta(delta, minDelta, maxDelta) for delta in deltas]
        
        # Draw molecules
        d = rdMolDraw2D.MolDraw2DCairo(300, 300)
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, 
            mol1, 
            highlightAtoms=list(range(mol1.GetNumAtoms())), 
            highlightAtomColors=colors
        )
        if not os.path.exists(outputDir + '/' + reactionName):
            os.makedirs(outputDir + '/' + reactionName)
        d.WriteDrawingText(outputDir + '/' + reactionName + '/' + compound + '.png')

if (False):
    # Load Reaction graph
    testGraph = None
    with open('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/data/graphs_with_master_node/R07308', 'rb') as file:
        testGraph = pickle.load(file)
    file.close()
    print(testGraph)

    node_feat_dim = testGraph.ndata['feat'].size()[-1]
    edge_feat_dim = testGraph.edata['feat'].size()[-1]
    out_dim = n_types

    # Load model
    model = DeeperGCN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hid_dim=256,
        out_dim=out_dim,
        num_layers=7,
        dropout=0.2,
        learn_beta=True
    ).to(device)
    model.load_state_dict(torch.load('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/model/tmpmodel.pt', map_location=torch.device('cpu')))
    model.eval()

    # Remove specific nodes and edges
    #testGraph = dgl.remove_nodes(testGraph, [11, 15])

    # Test model
    ps = model(testGraph, testGraph.edata['feat'], testGraph.ndata['feat'])
    print(torch.exp(ps))
    print(torch.exp(ps).topk(1, dim=1))

    # Node significance
    original_prob = 5319.6997
    removed_probs = [3941.7529, 1728.9265, 7637.9722, 5430.9053, 5903.3496, 9545.0254, 5201.6157, 0.87585, 23178.2598, 10186.2119, 10186.2119, 3232.8752, 5319.6997, 5319.6997, 5319.6997, 5319.6997] 
    removed_deltas = [((original_prob -  x) / original_prob) for x in removed_probs]
    print(removed_deltas)

    # Networkx visualization
    nxG = nx.Graph(dgl.to_networkx(testGraph))
    nx.draw(nxG, with_labels=True, cmap=plt.cm.Blues)#, node_color=removed_deltas)
    plt.show()

    # Rdkit visualization
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    import colorsys

    def getColorFromDelta(delta, minDelta, maxDelta):
        if (delta < 0):
            hue=0
            saturation = 1
            brightness = delta/minDelta
        else:
            hue=0.43
            saturation = 1
            brightness = delta/maxDelta

        a=0
        b=1
        c=1
        d=0.5
        brightness = c + (((d-c)/(b-a)) * (brightness-a))

        return colorsys.hls_to_rgb(hue, brightness, saturation)    

    def getAtomColors(mapping):
        colors = {}
        for idx in range(len(mapping)):
            colors[idx] = getColorFromDelta(removed_deltas[mapping[idx]], removed_deltas)
        return colors

    mol1 = Chem.MolFromMolFile('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsComplete/C01010.mol')
    mol1AtomMapping = [0, 1, 2, 3, 4, 5, 6] #Maps from rdkit atom indexes to removed_deltas
    d = rdMolDraw2D.MolDraw2DCairo(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, 
        mol1, 
        highlightAtoms=list(range(mol1.GetNumAtoms())), 
        highlightAtomColors=getAtomColors(mol1AtomMapping)
    )
    d.WriteDrawingText('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/images/temp1.png')

    mol2 = Chem.MolFromMolFile('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsComplete/C00001.mol')
    mol2AtomMapping = [7]
    d = rdMolDraw2D.MolDraw2DCairo(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, 
        mol2, 
        highlightAtoms=list(range(mol2.GetNumAtoms())), 
        highlightAtomColors=getAtomColors(mol2AtomMapping)
    )
    d.WriteDrawingText('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/images/temp2.png')

    mol3 = Chem.MolFromMolFile('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsComplete/C00011.mol')
    mol3AtomMapping = [8,9,10]
    d = rdMolDraw2D.MolDraw2DCairo(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, 
        mol3, 
        highlightAtoms=list(range(mol3.GetNumAtoms())), 
        highlightAtomColors=getAtomColors(mol3AtomMapping)
    )
    d.WriteDrawingText('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/images/temp3.png')

    mol4 = Chem.MolFromMolFile('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsComplete/C00014.mol')
    mol4AtomMapping = [11]
    d = rdMolDraw2D.MolDraw2DCairo(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, 
        mol4, 
        highlightAtoms=list(range(mol4.GetNumAtoms())), 
        highlightAtomColors=getAtomColors(mol4AtomMapping)
    )
    d.WriteDrawingText('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/images/temp4.png')

####################################

evaluateModel(
    modelPath='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/model/tmpmodel.pt',
    graphsDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/data/graphs_with_master_node',
    reactionName='R00005',
    reactionsDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions',
    csvDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/csv/csvAll',
    molDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols',
    outputDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/images'
)