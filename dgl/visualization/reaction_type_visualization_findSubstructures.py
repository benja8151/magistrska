import dgl
import dgl.function as dglfn
from dgl.data import DGLDataset
import os
from dgl.data.graph_serialize import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader, neighbor
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.functional import edge_softmax
from dgl.readout import mean_nodes
from numpy.core.fromnumeric import mean
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc
import copy
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from operator import itemgetter

import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
import colorsys
import math

import svg_stack as ss
import svgwrite
import cairo

n_types = 8
batch_size = 1
device = 'cpu'

reactionTypes = {
    0: "Hydrolase",
    1: "Isomerase",
    2: "Ligase",
    3: "Lyase",
    4: "Oxidoreductase",
    5: "Transferase",
    6: "Translocase",
    7: "Unassigned",
}

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
        with g.local_scope():
            hv = self.node_encoder(node_feats)
            #print("hv: ", hv.shape)
            he = edge_feats
            #print("he: ", he.shape)

            for layer in range(self.num_layers):
                #print("Layer: ", layer)
                hv1 = self.norms[layer](hv)
                #print("Normalization: ", hv1.shape)
                hv1 = F.relu(hv1)
                #print("Relu: ", hv1.shape)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                #print("Dropout: ", hv1.shape)
                hv = self.gcns[layer](g, hv1, he) + hv
                #print("Layer output: ", hv.shape)

            h_g = self.pooling(g, hv)
            #print("Pooling: ", h_g.shape)
            
            return self.output(h_g)

class RemovedDelta:
    def __init__(self, compoundName: str, atomIndex: int, delta: float):
        self.compoundName = compoundName
        self.atomIndex = atomIndex
        self.delta = delta

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

# Maps from rdkit atom indexes to DGL graph indexes  
# Currently only for graphs with master nodes  
# Returns tuple: (atomMappings, numReactants)
def createAtomMappings(reactionsDir: str, reactionName: str, csvDir: str):
    reactionFile = open(reactionsDir + '/' + reactionName)
    reactionSides = reactionFile.read().split('-')
    reactionFile.close()
    leftSide = reactionSides[0].split(',')
    rightSide = reactionSides[1].split(',')

    atomIndex = 0
    atomMapping = {}

    compoundOrder = []

    for compound in leftSide:
        nodes = pd.read_csv(csvDir + '/' + compound + '/nodes.csv')
        atomMapping[compound] = list(range(atomIndex, len(nodes)+atomIndex))
        atomIndex += len(nodes)
        compoundOrder.append(compound)

    for compound in rightSide:
        nodes = pd.read_csv(csvDir + '/' + compound + '/nodes.csv')
        if compound in atomMapping:
            compound = compound + "_2"
        atomMapping[compound] = list(range(atomIndex, len(nodes)+atomIndex))
        atomIndex += len(nodes)
        compoundOrder.append(compound)

    for compound in compoundOrder:
        atomMapping[compound] = atomMapping[compound] + [atomIndex]
        atomIndex+=1

    return atomMapping, len(leftSide), compoundOrder

# Runs evaluation on graph and returns class probability
def testModelOnGraph(model, testGraph, reactionType, testPrediction=False):
    ps = model(testGraph, testGraph.edata['feat'], testGraph.ndata['feat'])
    _, top_class = torch.exp(ps).topk(1, dim=1)
    if (testPrediction and top_class != reactionType):
        print("Incorrect prediction")
        raise Exception("Model did not predict correctly!")
    return torch.exp(ps)[0][reactionType].item()

# Opens reaction graph
def getOriginalReactionGraph(graphsDir: str, reactionName: str):
    graph = None
    with open(graphsDir + '/' + reactionName, 'rb') as file:
        graph = pickle.load(file)
    file.close()
    return graph

# Global variables to store structures
positiveStructures = {}
negativeStructures = {}

# Perfoms a series of model evaluations by removing single atoms
def evaluateModel(
    modelPath: str, 
    graphsDir: str, 
    reactionName: str, 
    reactionType: int,
    reactionsDir: str, 
    csvDir: str,
    molDir: str,
    skipIfIncorrectPrediction: bool
):
    
    # Load Reaction graph
    testGraph = getOriginalReactionGraph(graphsDir, reactionName)
    
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
    atomMappings, numReactants, compoundOrder = createAtomMappings(reactionsDir, reactionName, csvDir)

    # Calculate original graph evaluation
    originalProb = testModelOnGraph(model, testGraph, reactionType, skipIfIncorrectPrediction)

    # Evaluate classification when single atom is removed
    removedDeltas = []
    for compound, mapping in atomMappings.items():
        
        # Compounds with 1 atom and 1 master node - remove both
        if len(mapping) == 2:
            modifiedGraph = dgl.remove_nodes(getOriginalReactionGraph(graphsDir, reactionName), mapping)
            removedProb = testModelOnGraph(model, modifiedGraph, reactionType)
            removedDeltas.append(
                RemovedDelta(
                    compound,
                    0, 
                    (originalProb -  removedProb) / originalProb
                )
            )
        # Larger compounds
        else:
            for i in range(len(mapping) - 1):
                node = mapping[i]
                modifiedGraph = dgl.remove_nodes(getOriginalReactionGraph(graphsDir, reactionName), node)
                removedProb = testModelOnGraph(model, modifiedGraph, reactionType)
                removedDeltas.append(
                    RemovedDelta(
                        compound,
                        i,
                        (originalProb -  removedProb) / originalProb
                    )
                )

    # Find biggest/smallest deltas
    removedDeltas.sort(key=lambda x: x.delta)
    negativeDeltas = [] # bad for prediction
    positiveDeltas = [] # good for prediction
    for removedDelta in removedDeltas:
        if removedDelta.delta<0 and len(negativeDeltas)<3:
            negativeDeltas.append(removedDelta)
        else:
            break
    for removedDelta in reversed(removedDeltas):
        if removedDelta.delta>0 and len(positiveDeltas)<3:
            positiveDeltas.append(removedDelta)
        else:
            break

    # Add structures to dictionary
    for removedDelta in negativeDeltas:

        # Load .mol file
        mol = Chem.MolFromMolFile(molDir + '/' + removedDelta.compoundName.partition("_")[0] + '.mol', removeHs=False)

        # Find atom bonds
        bondsAll = mol.GetBonds()
        atomBonds = []
        for bond in bondsAll:
            if (bond.GetBeginAtomIdx() == removedDelta.atomIndex or bond.GetEndAtomIdx() == removedDelta.atomIndex):
                atomBonds.append(bond.GetIdx())

        # Create substructure
        struct = Chem.PathToSubmol(mol, atomBonds)

        # Find if substructure exists
        structExists = False
        for molKey in negativeStructures.keys():
            if (molKey.HasSubstructMatch(struct) and struct.HasSubstructMatch(molKey)):
                negativeStructures[molKey].append(removedDelta.delta)
                structExists = True
                break
        if not structExists:
            negativeStructures[struct] = [removedDelta.delta]


    for removedDelta in positiveDeltas:
        mol = Chem.MolFromMolFile(molDir + '/' + removedDelta.compoundName.partition("_")[0] + '.mol', removeHs=False)
        bondsAll = mol.GetBonds()
        atomBonds = []
        for bond in bondsAll:
            if (bond.GetBeginAtomIdx() == removedDelta.atomIndex or bond.GetEndAtomIdx() == removedDelta.atomIndex):
                atomBonds.append(bond.GetIdx())
        struct = Chem.PathToSubmol(mol, atomBonds)
        structExists = False
        for molKey in positiveStructures.keys():
            if (molKey.HasSubstructMatch(struct) and struct.HasSubstructMatch(molKey)):
                positiveStructures[molKey].append(removedDelta.delta)
                structExists = True
                break
        if not structExists:
            positiveStructures[struct] = [removedDelta.delta]


def findSubstructures(
    modelPath: str, 
    graphsDir: str,
    reactionType: int,
    reactionsDir: str, 
    typesDir: str,
    csvDir: str,
    molDir: str,
    tempImagesDir: str,
    outputDir: str,
    skipIfIncorrectPrediction: bool = False
):
    reactionNames = open(typesDir + "/" + reactionTypes[reactionType], 'r')
    checkedCount = 0

    # Evaluate all reactions of this type
    for reaction in reactionNames:
        print(checkedCount)
        try:
            evaluateModel(
                modelPath = modelPath,
                graphsDir = graphsDir,
                reactionName = reaction.strip(),
                reactionType = reactionType,
                reactionsDir = reactionsDir, 
                csvDir = csvDir,
                molDir = molDir,
                skipIfIncorrectPrediction = skipIfIncorrectPrediction
            )
            checkedCount += 1
        except:
            continue
    
    # Find most common substructures
    sortedNegativeStructures = sorted(negativeStructures.items(), key=lambda x: (len(x[1]), mean(x[1]) * -1), reverse=True)
    sortedPositiveStructures = sorted(positiveStructures.items(), key=lambda x: (len(x[1]), mean(x[1])), reverse=True)
    
    negativeStructuresTotalCount = 0
    positiveStructuresTotalCount = 0
    for _, deltas in sortedNegativeStructures:
        negativeStructuresTotalCount += len(deltas)
    for _, deltas in sortedPositiveStructures:
        positiveStructuresTotalCount += len(deltas)

    # Visualize substructures (5 for positive and negative)
    for index, (mol, deltas) in enumerate(sortedNegativeStructures[:5]):
        # Get Molecule Drawing Size
        d = rdMolDraw2D.MolDraw2DSVG(10000, 10000)
        d.drawOptions().fixedBondLength = 25
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        minX, minY, maxX, maxY = 10000, 10000, 0, 0
        for i in range(mol.GetNumAtoms()):
            coords = d.GetDrawCoords(i)
            minX = min(minX, coords.x)
            minY = min(minY, coords.y)
            maxX = max(maxX, coords.x)
            maxY = max(maxY, coords.y)

        padding = 100
        width = math.ceil(maxX - minX + padding)
        height = math.ceil(maxY - minY + padding)

        # Draw molecules
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawOptions = d.drawOptions()
        drawOptions.fixedBondLength = 25
        drawOptions.centreMoleculesBeforeDrawing = True
        rdMolDraw2D.MolDraw2D.SetDrawOptions(d, drawOptions)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        if not os.path.exists(tempImagesDir + "/" + reactionTypes[reactionType]):
            os.makedirs(tempImagesDir + "/" + reactionTypes[reactionType])
        f = open(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "negativeStruct_" + str(index) + '.svg', 'w')
        f.write(d.GetDrawingText())
        f.close()

        # Draw data
        textToWrite = [
            'SMILES: ' + Chem.MolToSmiles(mol),
            f'{len(deltas)} occurences ({round(100*len(deltas)/negativeStructuresTotalCount, 2)}%)',
            f'Average impact on prediction: {round(mean(deltas), 3)}'
        ]
        surface = cairo.SVGSurface('temp.svg', 10000, 10000)
        cr = cairo.Context(surface)
        cr.select_font_face('Arial', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(15)
        maxWidth = 0
        for text in textToWrite:
            xbearing, ybearing, width, height, xadvance, yadvance = cr.text_extents(text)
            maxWidth = max(maxWidth, width)

        dwg = svgwrite.Drawing(
            tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "negativeStructData_" + str(index) + '.svg', 
            size=(maxWidth + 40, 80),
            profile='tiny'
        )
        dwg.add(dwg.text(
            textToWrite[0],
            insert=(20, 20),
            font_size='15px',
            fill='black',
            font_family="Arial"
        ))
        dwg.add(dwg.text(
            textToWrite[1],
            insert=(20, 40),
            font_size='15px',
            fill='black',
            font_family="Arial"
        ))
        dwg.add(dwg.text(
            textToWrite[2],
            insert=(20, 60),
            font_size='15px',
            fill='black',
            font_family="Arial"
        ))
        dwg.save()

    for index, (mol, deltas) in enumerate(sortedPositiveStructures[:5]):
        # Get Molecule Drawing Size
        d = rdMolDraw2D.MolDraw2DSVG(10000, 10000)
        d.drawOptions().fixedBondLength = 25
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        minX, minY, maxX, maxY = 10000, 10000, 0, 0
        for i in range(mol.GetNumAtoms()):
            coords = d.GetDrawCoords(i)
            minX = min(minX, coords.x)
            minY = min(minY, coords.y)
            maxX = max(maxX, coords.x)
            maxY = max(maxY, coords.y)
            print(coords.x, coords.y)

        print(Chem.MolToSmiles(mol))
        print(mol.GetNumAtoms())

        padding = 100
        width = math.ceil(maxX - minX + padding)
        height = math.ceil(maxY - minY + padding)

        # Draw molecules
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawOptions = d.drawOptions()
        drawOptions.fixedBondLength = 25
        drawOptions.centreMoleculesBeforeDrawing = True
        rdMolDraw2D.MolDraw2D.SetDrawOptions(d, drawOptions)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        if not os.path.exists(tempImagesDir + "/" + reactionTypes[reactionType]):
            os.makedirs(tempImagesDir + "/" + reactionTypes[reactionType])
        f = open(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "positiveStruct_" + str(index) + '.svg', 'w')
        f.write(d.GetDrawingText())
        f.close()

        # Draw data
        textToWrite = [
            'SMILES: ' + Chem.MolToSmiles(mol),
            f'{len(deltas)} occurences ({round(100*len(deltas)/positiveStructuresTotalCount, 2)}%)',
            f'Average impact on prediction: {round(mean(deltas), 3)}'
        ]
        surface = cairo.SVGSurface('temp.svg', 10000, 10000)
        cr = cairo.Context(surface)
        cr.select_font_face('Arial', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(15)
        maxWidth = 0
        for text in textToWrite:
            xbearing, ybearing, width, height, xadvance, yadvance = cr.text_extents(text)
            maxWidth = max(maxWidth, width)

        dwg = svgwrite.Drawing(
            tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "positiveStructData_" + str(index) + '.svg', 
            size=(maxWidth + 40, 80),
            profile='tiny'
        )
        dwg.add(dwg.text(
            textToWrite[0],
            insert=(20, 20),
            font_size='15px',
            fill='black',
            font_family="Arial"
        ))
        dwg.add(dwg.text(
            textToWrite[1],
            insert=(20, 40),
            font_size='15px',
            fill='black',
            font_family="Arial"
        ))
        dwg.add(dwg.text(
            textToWrite[2],
            insert=(20, 60),
            font_size='15px',
            fill='black',
            font_family="Arial"
        ))
        dwg.save()

    # Combine .svgs
    # Negative
    for i in range(min(len(sortedNegativeStructures), 5)):
        doc = ss.Document()
        layoutIndividual = ss.VBoxLayout()
        layoutIndividual.addSVG(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "negativeStruct_" + str(i) + '.svg', alignment=ss.AlignHCenter)
        layoutIndividual.addSVG(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "negativeStructData_" + str(i) + '.svg', alignment=ss.AlignHCenter)
        doc.setLayout(layoutIndividual)
        doc.save(tempImagesDir + '/' + reactionTypes[reactionType] + '/' + "negativeTemp_" + str(i) + '.svg')
    doc = ss.Document()
    layoutSeries = ss.HBoxLayout()
    for i in range(min(len(sortedNegativeStructures), 5)):
        layoutSeries.addSVG(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "negativeTemp_" + str(i) + '.svg', alignment=ss.AlignVCenter)
    doc.setLayout(layoutSeries)
    doc.save(tempImagesDir + '/' + reactionTypes[reactionType] + '/' + "negativeTemp" + '.svg')

    # Positive
    for i in range(min(len(sortedPositiveStructures), 5)):
        doc = ss.Document()
        layoutIndividual = ss.VBoxLayout()
        layoutIndividual.addSVG(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "positiveStruct_" + str(i) + '.svg', alignment=ss.AlignHCenter)
        layoutIndividual.addSVG(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "positiveStructData_" + str(i) + '.svg', alignment=ss.AlignHCenter)
        doc.setLayout(layoutIndividual)
        doc.save(tempImagesDir + '/' + reactionTypes[reactionType] + '/' + "positiveTemp_" + str(i) + '.svg')
    doc = ss.Document()
    layoutSeries = ss.HBoxLayout()
    for i in range(min(len(sortedPositiveStructures), 5)):
        layoutSeries.addSVG(tempImagesDir + "/" + reactionTypes[reactionType] + '/' + "positiveTemp_" + str(i) + '.svg', alignment=ss.AlignVCenter)
    doc.setLayout(layoutSeries)
    doc.save(tempImagesDir + '/' + reactionTypes[reactionType] + '/' + "positiveTemp" + '.svg')
    
    # Combined
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    layoutComplete = ss.VBoxLayout()
    layoutComplete.addSVG(tempImagesDir + '/positive.svg', alignment=ss.AlignLeft)
    layoutComplete.addSVG(tempImagesDir + '/' + reactionTypes[reactionType] + '/' + "positiveTemp" + '.svg', alignment=ss.AlignHCenter)
    layoutComplete.addSVG(tempImagesDir + '/negative.svg', alignment=ss.AlignLeft)
    layoutComplete.addSVG(tempImagesDir + '/' + reactionTypes[reactionType] + '/' + "negativeTemp" + '.svg', alignment=ss.AlignHCenter)
    doc.setLayout(layoutComplete)
    doc.save(outputDir + '/' + reactionTypes[reactionType] + '.svg')


####################################

""" 
    Reaction Types:
    0: Hydrolase
    1: Isomerase
    2: Ligase
    3: Lyase
    4: Oxidoreductase
    5: Transferase
    6: Translocase
    7: Unassigned
"""
findSubstructures(
    modelPath='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/model/tmpmodel.pt',
    graphsDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/data/graphs_with_master_node',
    reactionType=6,
    reactionsDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions',
    typesDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactionTypes',
    csvDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/csv/csvAll',
    molDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsComplete',
    tempImagesDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/images/structureFinding',
    outputDir='C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/results_structureFinding',
    skipIfIncorrectPrediction=True
)
