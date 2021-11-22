from typing import Any, Dict, List, Tuple
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

import xlsxwriter
import PIL

import midPointNorm as mpn

import seaborn as sns
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as sds
from sklearn.cluster import AgglomerativeClustering

n_types = 8
batch_size = 1
device = "cpu"

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
    def __init__(self, raw_dir: str, save_dir: str, labelsFile: str, force_reload=False):
        self.labelsFile = labelsFile
        super(ReactionsDataset, self).__init__(
            name="ReactionsDataset", raw_dir=raw_dir, save_dir=save_dir, verbose=True, force_reload=force_reload
        )

    def process(self):
        self.graphs = []
        self.labels = []

        file = pd.read_csv(self.labelsFile)
        reactions = file["Reaction"]
        types = file["Type"]

        print("Loading graphs...")
        count = 0
        for reaction, type in zip(reactions, types):

            # TODO: temp
            if count > 200:
                break

            g = self.loadGraph(self.raw_dir + "/" + reaction)
            if g is not None:
                self.graphs.append(g)
                self.labels.append(type)

            print(str(round((count / len(reactions) * 100), 2)) + "%")
            count += 1

        self.labels = torch.LongTensor(self.labels)

    def loadGraph(self, path: str):
        try:
            with open(path, "rb") as file:
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
        save_graphs(os.path.join(self.save_path, "dataset.bin"), self.graphs, {"labels": self.labels})

    def load(self):
        print("Loading dataset from cache...")
        graphs, labels = load_graphs(os.path.join(self.save_path, "dataset.bin"))
        self.graphs = graphs
        self.labels = labels["labels"]

    def has_cache(self):
        print("Checking if cached file exists...")
        return os.path.exists(os.path.join(self.save_path, "dataset.bin"))


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
            # torch.set_printoptions(threshold=10_000)
            # print(x[:,i].long())
            try:
                x_embedding += self.atom_embedding_list[i](x[:, i].long())
            except:
                torch.set_printoptions(threshold=10_000)
                print(x[:, i].long())
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
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i].long())

        return bond_embedding


# Neural Network
class MLP(nn.Sequential):
    r"""

    Description
    -----------
    From equation (5) in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    """

    def __init__(self, channels, act="relu", dropout=0.0, bias=True):
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

    def __init__(
        self,
        in_dim,
        out_dim,
        aggregator="softmax",
        beta=1.0,
        learn_beta=False,
        p=1.0,
        learn_p=False,
        msg_norm=False,
        learn_msg_scale=False,
        mlp_layers=1,
        eps=1e-7,
    ):
        super(GENConv, self).__init__()

        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = (
            nn.Parameter(torch.Tensor([beta]), requires_grad=True) if learn_beta and self.aggr == "softmax" else beta
        )
        self.p = nn.Parameter(torch.Tensor([p]), requires_grad=True) if learn_p else p

        self.edge_encoder = BondEncoder(in_dim)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # Node and edge feature size need to match.
            g.ndata["h"] = node_feats
            g.edata["h"] = self.edge_encoder(edge_feats)
            g.apply_edges(dglfn.u_add_e("h", "h", "m"))

            if self.aggr == "softmax":
                g.edata["m"] = F.relu(g.edata["m"]) + self.eps
                g.edata["a"] = edge_softmax(g, g.edata["m"] * self.beta)
                g.update_all(lambda edge: {"x": edge.data["m"] * edge.data["a"]}, dglfn.sum("x", "m"))

            elif self.aggr == "power":
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.edata["m"], minv, maxv)
                g.update_all(lambda edge: {"x": torch.pow(edge.data["m"], self.p)}, dglfn.mean("x", "m"))
                torch.clamp_(g.ndata["m"], minv, maxv)
                g.ndata["m"] = torch.pow(g.ndata["m"], self.p)

            else:
                raise NotImplementedError(f"Aggregator {self.aggr} is not supported.")

            if self.msg_norm is not None:
                g.ndata["m"] = self.msg_norm(node_feats, g.ndata["m"])

            feats = node_feats + g.ndata["m"]

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

    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout=0.0,
        beta=1.0,
        learn_beta=False,
        aggr="softmax",
        mlp_layers=1,
    ):
        super(DeeperGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            conv = GENConv(
                in_dim=hid_dim,
                out_dim=hid_dim,
                aggregator=aggr,
                beta=beta,
                learn_beta=learn_beta,
                mlp_layers=mlp_layers,
            )

            self.gcns.append(conv)
            self.norms.append(nn.BatchNorm1d(hid_dim, affine=True))

        self.node_encoder = AtomEncoder(hid_dim)
        self.pooling = AvgPooling()
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, edge_feats, node_feats=None):
        with g.local_scope():
            hv = self.node_encoder(node_feats)
            # print("hv: ", hv.shape)
            he = edge_feats
            # print("he: ", he.shape)

            for layer in range(self.num_layers):
                # print("Layer: ", layer)
                hv1 = self.norms[layer](hv)
                # print("Normalization: ", hv1.shape)
                hv1 = F.relu(hv1)
                # print("Relu: ", hv1.shape)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                # print("Dropout: ", hv1.shape)
                hv = self.gcns[layer](g, hv1, he) + hv
                # print("Layer output: ", hv.shape)

            h_g = self.pooling(g, hv)
            # print("Pooling: ", h_g.shape)

            return self.output(h_g)


# Train/Test functions
def train(model, device, data_loader, opt, loss_fn):
    model.train()

    train_loss = []
    for g, labels in data_loader:
        g = g.to(device)
        labels = labels.to(device)

        log_ps = model(g, g.edata["feat"], g.ndata["feat"])

        loss = loss_fn(log_ps, labels)
        train_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    return sum(train_loss) / len(train_loss)


@torch.no_grad()
def test(model, device, data_loader, calculate_auc):
    model.eval()

    actual_labels, predicted_labels, predicted_probabilities = (
        [[] for i in range(n_types)],
        [[] for i in range(n_types)],
        [[] for i in range(n_types)],
    )

    accuracy = 0

    if not calculate_auc:
        for g, labels in data_loader:
            g = g.to(device)
            log_ps = model(g, g.edata["feat"], g.ndata["feat"])
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.to(device).view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    else:
        # OneVsRest
        for i in range(n_types):
            for g, labels in data_loader:
                g = g.to(device)
                log_ps = model(g, g.edata["feat"], g.ndata["feat"])
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
            fpr, tpr, _ = roc_curve(actual_labels[i], predicted_probabilities[i])
            print("Type:", i, "Auc: ", auc(fpr, tpr))

    return accuracy / len(data_loader)


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
    reactionFile = open(reactionsDir + "/" + reactionName)
    reactionSides = reactionFile.read().split("-")
    reactionFile.close()
    leftSide = reactionSides[0].split(",")
    rightSide = reactionSides[1].split(",")

    atomIndex = 0
    atomMapping = {}

    compoundOrder = []

    for compound in leftSide:
        nodes = pd.read_csv(csvDir + "/" + compound + "/nodes.csv")
        atomMapping[compound] = list(range(atomIndex, len(nodes) + atomIndex))
        atomIndex += len(nodes)
        compoundOrder.append(compound)

    for compound in rightSide:
        nodes = pd.read_csv(csvDir + "/" + compound + "/nodes.csv")
        if compound in atomMapping:
            compound = compound + "_2"
        atomMapping[compound] = list(range(atomIndex, len(nodes) + atomIndex))
        atomIndex += len(nodes)
        compoundOrder.append(compound)

    for compound in compoundOrder:
        atomMapping[compound] = atomMapping[compound] + [atomIndex]
        atomIndex += 1

    return atomMapping, len(leftSide), compoundOrder


# Runs evaluation on graph and returns class probability
def testModelOnGraph(model, testGraph, reactionType, testPrediction=False):
    ps = model(testGraph, testGraph.edata["feat"], testGraph.ndata["feat"])
    _, top_class = torch.exp(ps).topk(1, dim=1)
    if testPrediction and top_class != reactionType:
        print("Incorrect prediction")
        raise Exception("Model did not predict correctly!")
    return torch.exp(ps)[0][reactionType].item()


# Opens reaction graph
def getOriginalReactionGraph(graphsDir: str, reactionName: str):
    graph = None
    with open(graphsDir + "/" + reactionName, "rb") as file:
        graph = pickle.load(file)
    file.close()
    return graph


class RemovedDelta:
    def __init__(self, compoundName: str, atomIndex: int, delta: float):
        self.compoundName = compoundName
        self.atomIndex = atomIndex
        self.delta = delta


# Perfoms a series of model evaluations by removing single atoms
def evaluateModel(
    modelPath: str,
    graphsDir: str,
    reactionName: str,
    reactionType: int,
    reactionsDir: str,
    csvDir: str,
    molDir: str,
    skipIfIncorrectPrediction: bool,
    useAbsoluteDeltas: bool,
):

    # Load Reaction graph
    testGraph = getOriginalReactionGraph(graphsDir, reactionName)

    node_feat_dim = testGraph.ndata["feat"].size()[-1]
    edge_feat_dim = testGraph.edata["feat"].size()[-1]
    out_dim = n_types

    # Load model
    model = DeeperGCN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hid_dim=256,
        out_dim=out_dim,
        num_layers=7,
        dropout=0.2,
        learn_beta=True,
    ).to(device)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device("cpu")))
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
            removedDeltaValue = (
                originalProb - removedProb if useAbsoluteDeltas else (originalProb - removedProb) / originalProb
            )
            removedDeltas.append(RemovedDelta(compound, 0, removedDeltaValue))
        # Larger compounds
        else:
            for i in range(len(mapping) - 1):
                node = mapping[i]
                modifiedGraph = dgl.remove_nodes(getOriginalReactionGraph(graphsDir, reactionName), node)
                removedProb = testModelOnGraph(model, modifiedGraph, reactionType)

                removedDeltaValue = (
                    originalProb - removedProb if useAbsoluteDeltas else (originalProb - removedProb) / originalProb
                )

                removedDeltas.append(RemovedDelta(compound, i, removedDeltaValue))

    # Find biggest/smallest deltas
    removedDeltas.sort(key=lambda x: x.delta)
    negativeDeltas = []  # bad for prediction
    positiveDeltas = []  # good for prediction
    for removedDelta in removedDeltas:
        if removedDelta.delta < 0 and len(negativeDeltas) < 3:
            negativeDeltas.append(removedDelta)
        else:
            break
    for removedDelta in reversed(removedDeltas):
        if removedDelta.delta > 0 and len(positiveDeltas) < 3:
            positiveDeltas.append(removedDelta)
        else:
            break

    return negativeDeltas, positiveDeltas


def findSubstructuresForReactionType(
    modelPath: str,
    graphsDir: str,
    reactionType: int,
    reactionsDir: str,
    typesDir: str,
    csvDir: str,
    molDir: str,
    tempImagesDir: str,
    outputDir: str,
    skipIfIncorrectPrediction: bool = False,
    useAbsoluteDeltas: bool = True,
) -> Tuple[List[Tuple[Any, List[float]]], List[Tuple[Any, List[float]]]]:
    """Returns a tuple (sortedNegativeStructures, sortedPositiveStructures)
    Return type: Tuple(List(Tuple(mol, List(float))), List(Tuple(mol, List(float))))"""
    reactionNames = open(typesDir + "/" + reactionTypes[reactionType], "r")
    checkedCount = 0

    # Variables to store structures
    positiveStructures = {}
    negativeStructures = {}

    # Evaluate all reactions of this type
    for reaction in reactionNames:
        print(checkedCount)

        # TODO: temp
        # if checkedCount > 10:
        #    break

        try:
            negativeDeltas, positiveDeltas = evaluateModel(
                modelPath=modelPath,
                graphsDir=graphsDir,
                reactionName=reaction.strip(),
                reactionType=reactionType,
                reactionsDir=reactionsDir,
                csvDir=csvDir,
                molDir=molDir,
                skipIfIncorrectPrediction=skipIfIncorrectPrediction,
                useAbsoluteDeltas=useAbsoluteDeltas,
            )

            # Add structures to dictionary
            for removedDelta in negativeDeltas:

                # Load .mol file
                mol = Chem.MolFromMolFile(
                    molDir + "/" + removedDelta.compoundName.partition("_")[0] + ".mol", removeHs=False
                )

                # Find atom bonds and check if it's a singular atom
                bondsAll = mol.GetBonds()
                if len(bondsAll) == 0:
                    struct = mol
                else:
                    atomBonds = []
                    for bond in bondsAll:
                        if (
                            bond.GetBeginAtomIdx() == removedDelta.atomIndex
                            or bond.GetEndAtomIdx() == removedDelta.atomIndex
                        ):
                            atomBonds.append(bond.GetIdx())

                    # Create substructure
                    struct = Chem.PathToSubmol(mol, atomBonds)

                # Find if substructure exists
                structExists = False
                for molKey in negativeStructures.keys():
                    if molKey.HasSubstructMatch(struct) and struct.HasSubstructMatch(molKey):
                        negativeStructures[molKey].append(removedDelta.delta)
                        structExists = True
                        break
                if not structExists:
                    negativeStructures[struct] = [removedDelta.delta]

            for removedDelta in positiveDeltas:
                mol = Chem.MolFromMolFile(
                    molDir + "/" + removedDelta.compoundName.partition("_")[0] + ".mol", removeHs=False
                )
                bondsAll = mol.GetBonds()
                if len(bondsAll) == 0:
                    struct = mol
                else:
                    atomBonds = []
                    for bond in bondsAll:
                        if (
                            bond.GetBeginAtomIdx() == removedDelta.atomIndex
                            or bond.GetEndAtomIdx() == removedDelta.atomIndex
                        ):
                            atomBonds.append(bond.GetIdx())
                    struct = Chem.PathToSubmol(mol, atomBonds)
                structExists = False
                for molKey in positiveStructures.keys():
                    if molKey.HasSubstructMatch(struct) and struct.HasSubstructMatch(molKey):
                        positiveStructures[molKey].append(removedDelta.delta)
                        structExists = True
                        break
                if not structExists:
                    positiveStructures[struct] = [removedDelta.delta]

            # Find most common substructures
            sortedNegativeStructures = sorted(
                negativeStructures.items(), key=lambda x: (len(x[1]), mean(x[1]) * -1), reverse=True
            )
            sortedPositiveStructures = sorted(
                positiveStructures.items(), key=lambda x: (len(x[1]), mean(x[1])), reverse=True
            )

            checkedCount += 1
        except:
            continue

    return sortedNegativeStructures, sortedPositiveStructures


# Find maximum and minimum deltas for specific structure and return Norm
def findStructureMaxMinDeltas(filteredStructures: Dict):
    results = {}
    for mol, data in filteredStructures.items():
        maxDelta = 1
        minDelta = -1
        for reactionTypeData in data.values():
            if "pos" in reactionTypeData:
                maxDelta = max(maxDelta, reactionTypeData["pos"].averageDelta)
            if "neg" in reactionTypeData:
                minDelta = min(minDelta, reactionTypeData["neg"].averageDelta)
        results[mol] = mpn.MidPointNorm(
            midpoint=0,
            vmin=minDelta,
            vmax=maxDelta,
        )
    return results


class StructureDeltaResult:
    def __init__(self, deltas: List[float], reactionType: int):
        self.averageDelta = mean(deltas)
        self.count = len(deltas)
        self.calculateMaxMin(reactionType)

    def calculateMaxMin(self, reactionType: int):
        if self.averageDelta > 0:
            maxMinDeltas[reactionType]["max"] = max(maxMinDeltas[reactionType]["max"], self.averageDelta)
        else:
            maxMinDeltas[reactionType]["min"] = min(maxMinDeltas[reactionType]["min"], self.averageDelta)


class WeightedStructureDeltaResult:
    def __init__(self, averageDelta: float, count: int):
        self.averageDelta = averageDelta
        self.count = count
        self.countPercentage = 0

    def calculateWeightedScore(self, totalStructureOccurences: int, structureMaxDelta: float, structureMinDelta: float):
        if self.averageDelta == 0:
            return

        self.countPercentage = self.count / totalStructureOccurences

        if self.averageDelta > 0:
            self.averageDelta = self.averageDelta * self.countPercentage / structureMaxDelta
        else:
            self.averageDelta = self.averageDelta * self.countPercentage / structureMinDelta * -1


def getColorFromDelta(delta, cmap, norm):
    return mpl.colors.rgb2hex(cmap(norm(delta)))


def compareReactionsBySubstructures(
    modelPath: str,
    graphsDir: str,
    reactionsDir: str,
    typesDir: str,
    csvDir: str,
    molDir: str,
    tempImagesDir: str,
    outputDir: str,
    outputName: str,
    skipIfIncorrectPrediction: bool = False,
    useAbsoluteDeltas: bool = True,
    performCleanEvaluation: bool = True,
    evaluatedReactionTypes: list = list(range(7)),
):

    # Combine and compare structures accross reaction types
    filteredStructures = {}

    # Dictionary to store maximum and minimum deltas for reaction type
    global maxMinDeltas
    maxMinDeltas = {reactionType: {"max": 1, "min": -1} for reactionType in reactionTypes}

    didRetrieveResults = False

    # Try to load data from saved dictionaries
    if not performCleanEvaluation:
        try:
            with open(outputDir + "/filteredStructures_dict.pickle", "rb") as handle:
                filteredStructures = pickle.load(handle)
            with open(outputDir + "/maxMinDeltas_dict.pickle", "rb") as handle:
                maxMinDeltas = pickle.load(handle)
            didRetrieveResults = True
            print("Retrieved saved data from dictionaries.")
        except:
            print("Failed to retrieve saved data from dictionaries, performing clean evaluation.")

    if not didRetrieveResults:
        combinedNegativeStructures = {}
        combinedPositiveStructures = {}

        # Get all structures for every reaction type
        for reactionType in evaluatedReactionTypes:
            print("######################################################")
            print(f"Finding structures for: {reactionTypes[reactionType]}")
            print("######################################################")
            negativeStructures, positiveStructures = findSubstructuresForReactionType(
                modelPath=modelPath,
                graphsDir=graphsDir,
                reactionType=reactionType,
                reactionsDir=reactionsDir,
                typesDir=typesDir,
                csvDir=csvDir,
                molDir=molDir,
                tempImagesDir=tempImagesDir,
                outputDir=outputDir,
                skipIfIncorrectPrediction=skipIfIncorrectPrediction,
                useAbsoluteDeltas=useAbsoluteDeltas,
            )

            combinedNegativeStructures[reactionType] = negativeStructures
            combinedPositiveStructures[reactionType] = positiveStructures

        # Negative (top 10)
        for reactionType, typeStructures in combinedNegativeStructures.items():
            structuresCount = 0
            while structuresCount < min(5, len(typeStructures)):
                structureMol, structureDeltas = typeStructures[structuresCount]

                # Find if substructure already exists
                structExists = False
                for molKey in filteredStructures.keys():
                    if molKey.HasSubstructMatch(structureMol) and structureMol.HasSubstructMatch(molKey):

                        # Check for structure doubling for each reaction type
                        if reactionType in filteredStructures[molKey]:
                            filteredStructures[molKey][reactionType]["neg"] = StructureDeltaResult(
                                structureDeltas, reactionType
                            )
                        else:
                            filteredStructures[molKey][reactionType] = {
                                "neg": StructureDeltaResult(structureDeltas, reactionType)
                            }
                        structExists = True
                        break
                if not structExists:
                    filteredStructures[structureMol] = {
                        reactionType: {"neg": StructureDeltaResult(structureDeltas, reactionType)}
                    }

                structuresCount += 1
        # Positive (top 10)
        for reactionType, typeStructures in combinedPositiveStructures.items():
            structuresCount = 0
            while structuresCount < min(5, len(typeStructures)):
                structureMol, structureDeltas = typeStructures[structuresCount]

                # Find if substructure already exists
                structExists = False
                for molKey in filteredStructures.keys():
                    if molKey.HasSubstructMatch(structureMol) and structureMol.HasSubstructMatch(molKey):

                        # Check for structure doubling for each reaction type
                        if reactionType in filteredStructures[molKey]:
                            filteredStructures[molKey][reactionType]["pos"] = StructureDeltaResult(
                                structureDeltas, reactionType
                            )
                        else:
                            filteredStructures[molKey][reactionType] = {
                                "pos": StructureDeltaResult(structureDeltas, reactionType)
                            }
                        structExists = True
                        break
                if not structExists:
                    filteredStructures[structureMol] = {
                        reactionType: {"pos": StructureDeltaResult(structureDeltas, reactionType)}
                    }

                structuresCount += 1

        # Additional checks (non top-n most frequent structures)
        for mol, data in filteredStructures.items():
            for reactionType in evaluatedReactionTypes:
                if reactionType in data:
                    if "neg" not in data[reactionType]:
                        for structureMol, structureDeltas in combinedNegativeStructures[reactionType]:
                            if mol.HasSubstructMatch(structureMol) and structureMol.HasSubstructMatch(mol):
                                data[reactionType]["neg"] = StructureDeltaResult(structureDeltas, reactionType)
                    if "pos" not in data[reactionType]:
                        for structureMol, structureDeltas in combinedPositiveStructures[reactionType]:
                            if mol.HasSubstructMatch(structureMol) and structureMol.HasSubstructMatch(mol):
                                data[reactionType]["pos"] = StructureDeltaResult(structureDeltas, reactionType)
                else:
                    tempData = {}
                    for structureMol, structureDeltas in combinedNegativeStructures[reactionType]:
                        if mol.HasSubstructMatch(structureMol) and structureMol.HasSubstructMatch(mol):
                            tempData["neg"] = StructureDeltaResult(structureDeltas, reactionType)
                    for structureMol, structureDeltas in combinedPositiveStructures[reactionType]:
                        if mol.HasSubstructMatch(structureMol) and structureMol.HasSubstructMatch(mol):
                            tempData["pos"] = StructureDeltaResult(structureDeltas, reactionType)
                    data[reactionType] = tempData

        # TODO Save Filtered Structures Dictionary and deltas for possible later use
        print("Saving filtered structures dictionary and deltas...")
        with open(outputDir + "/filteredStructures_dict.pickle", "wb") as handle:
            pickle.dump(filteredStructures, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(outputDir + "/maxMinDeltas_dict.pickle", "wb") as handle:
            pickle.dump(maxMinDeltas, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Successfully saved!")

    # Calculate weighted scores (delta * occurences / combined occurences)
    weightedFilteredStructures = {}
    for mol, data in filteredStructures.items():
        totalStructureOccurences = 0
        structureData = {}

        structureMaxDelta = 0
        structureMinDelta = 0

        for reactionType in evaluatedReactionTypes:
            reactionTypeData = data[reactionType]
            reactionTypeScore = 0
            reactionTypeCount = 0
            if "neg" in reactionTypeData:
                structureCount = reactionTypeData["neg"].count
                totalStructureOccurences += structureCount
                reactionTypeScore += reactionTypeData["neg"].averageDelta * structureCount
                reactionTypeCount += structureCount
            if "pos" in reactionTypeData:
                structureCount = reactionTypeData["pos"].count
                totalStructureOccurences += structureCount
                reactionTypeScore += reactionTypeData["pos"].averageDelta * structureCount
                reactionTypeCount += structureCount

            structureMaxDelta = max(structureMaxDelta, reactionTypeScore)
            structureMinDelta = min(structureMinDelta, reactionTypeScore)

            # if reactionTypeCount > 0:
            #    reactionTypeScore = reactionTypeScore / reactionTypeCount

            structureData[reactionType] = WeightedStructureDeltaResult(reactionTypeScore, reactionTypeCount)

        # structureMaxDelta /= totalStructureOccurences
        # tructureMinDelta /= totalStructureOccurences

        for weightedStructureDeltaResult in structureData.values():
            weightedStructureDeltaResult.calculateWeightedScore(
                totalStructureOccurences, structureMaxDelta, structureMinDelta
            )
        weightedFilteredStructures[mol] = structureData

    # Hierarchical clustering
    dataframeDict = {}
    dataFrameArray = []
    molIds = {}
    count = 0
    for mol, data in weightedFilteredStructures.items():
        molId = f"mol_{count}"
        molIds[molId] = mol
        molData = []
        for reactionType in evaluatedReactionTypes:
            molData.append(data[reactionType].averageDelta)
        dataframeDict[molId] = molData
        dataFrameArray.append(molData)
        count += 1
    dataFrame = pd.DataFrame.from_dict(
        dataframeDict, orient="index", columns=[reactionTypes[i] for i in evaluatedReactionTypes]
    )

    clusteringMethod = "ward"

    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    dend = shc.dendrogram(
        shc.linkage(dataFrame, method=clusteringMethod, optimal_ordering=True),
        labels=[Chem.MolToSmiles(molIds[f"mol_{molId}"]) for molId in range(count)],
    )
    plt.xticks(rotation=90)

    # print("My linkage")
    # print(shc.linkage(dataFrame.T, method=clusteringMethod)) # Columns Linkage
    # print(shc.linkage(dataFrame, method=clusteringMethod))  # Rows Linkage

    # print(dend)

    # cluster = AgglomerativeClustering(n_clusters=len(evaluatedReactionTypes), affinity="euclidean", linkage="ward")
    # cluster.fit_predict(dataFrameArray)

    g = sns.clustermap(dataFrameArray, method=clusteringMethod, cmap="PiYG")
    sortedData = g.data2d
    sortedRows = sortedData.index
    sortedColumns = sortedData.columns

    g.ax_heatmap.set_xticklabels([reactionTypes[reactionType] for reactionType in sortedColumns])
    g.ax_heatmap.set_yticklabels([Chem.MolToSmiles(molIds[f"mol_{molId}"]) for molId in sortedRows])
    g.ax_heatmap.set_title("Clustermap")

    plt.show()

    return

    # Calculated weighted norms
    weightedStructuresNorms = {}
    for mol, data in weightedFilteredStructures.items():
        maxDelta = 1
        minDelta = -1
        for reactionTypeData in data.values():
            maxDelta = max(maxDelta, reactionTypeData.averageDelta)
            minDelta = min(minDelta, reactionTypeData.averageDelta)
        weightedStructuresNorms[mol] = mpn.MidPointNorm(
            midpoint=0,
            vmin=minDelta,
            vmax=maxDelta,
        )

    # Prepare substructures images
    mol_id = 0
    for mol in filteredStructures.keys():
        # Get Molecule Drawing Size
        d = rdMolDraw2D.MolDraw2DCairo(10000, 10000)
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

        padding = 40
        width = math.ceil(maxX - minX + padding)
        height = math.ceil(maxY - minY + padding)

        if width < 1:
            width = 100
        if height < 1:
            height = 100

        # Draw molecules
        d = rdMolDraw2D.MolDraw2DCairo(width, height)
        if mol.GetNumAtoms() > 1:
            drawOptions = d.drawOptions()
            drawOptions.fixedBondLength = 25
            drawOptions.centreMoleculesBeforeDrawing = True
            rdMolDraw2D.MolDraw2D.SetDrawOptions(d, drawOptions)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        if not os.path.exists(tempImagesDir):
            os.makedirs(tempImagesDir)
        d.WriteDrawingText(tempImagesDir + "/" + "molStruct_" + str(mol_id) + ".png")

        mol_id += 1

    # Create comparison table
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    workbook = xlsxwriter.Workbook(outputDir + "/" + outputName + ".xlsx")

    # Create 2 sheets - different color comparison
    # Sheet 1 - color relative to reaction type deltas
    # Sheet 2 - color relative to structure deltas
    worksheet1 = workbook.add_worksheet("Color by Reaction Type")
    worksheet2 = workbook.add_worksheet("Color by Structure")
    # Additional 2 sheets for weighted comparison and clustering of structures
    worksheet3 = workbook.add_worksheet("Weighted Color by Structure")
    worksheet4 = workbook.add_worksheet("Clustered Structures")

    for i in evaluatedReactionTypes:
        worksheet1.write(0, i + 2, reactionTypes[i])
        worksheet2.write(0, i + 2, reactionTypes[i])
        worksheet3.write(0, i + 2, reactionTypes[i])
        worksheet4.write(0, i + 2, reactionTypes[sortedColumns[i]])

    # Header
    worksheet1.write("A1", "Structure")
    worksheet2.write("A1", "Structure")
    worksheet3.write("A1", "Structure")
    worksheet4.write("A1", "Structure")
    worksheet1.write("B1", "SMILES")
    worksheet2.write("B1", "SMILES")
    worksheet3.write("B1", "SMILES")
    worksheet4.write("B1", "SMILES")
    worksheet1.set_column_pixels(
        "B:B", 160, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "right": 2})
    )
    worksheet2.set_column_pixels(
        "B:B", 160, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "right": 2})
    )
    worksheet3.set_column_pixels(
        "B:B", 160, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "right": 2})
    )
    worksheet4.set_column_pixels(
        "B:B", 160, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "right": 2})
    )
    worksheet1.set_row_pixels(
        0, 26, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "bottom": 2})
    )
    worksheet2.set_row_pixels(
        0, 26, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "bottom": 2})
    )
    worksheet3.set_row_pixels(
        0, 26, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "bottom": 2})
    )
    worksheet4.set_row_pixels(
        0, 26, cell_format=workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "bottom": 2})
    )
    worksheet1.freeze_panes(1, 0)
    worksheet2.freeze_panes(1, 0)
    worksheet3.freeze_panes(1, 0)
    worksheet4.freeze_panes(1, 0)

    # Columns formatting
    worksheet1.set_column_pixels(2, len(evaluatedReactionTypes) + 1, 160)
    worksheet2.set_column_pixels(2, len(evaluatedReactionTypes) + 1, 160)
    worksheet3.set_column_pixels(2, len(evaluatedReactionTypes) + 1, 160)
    worksheet4.set_column_pixels(2, len(evaluatedReactionTypes) + 1, 160)

    # Results
    mol_id = 0
    maxImageWidth = 80
    minRowHeight = 80
    padding = 4
    for i in range(len(filteredStructures)):
        imageWidth, _ = PIL.Image.open(tempImagesDir + "/molStruct_" + str(i) + ".png").size
        maxImageWidth = max(maxImageWidth, imageWidth + padding)
    worksheet1.set_column_pixels(0, 0, maxImageWidth)
    worksheet2.set_column_pixels(0, 0, maxImageWidth)
    worksheet3.set_column_pixels(0, 0, maxImageWidth)
    worksheet4.set_column_pixels(0, 0, maxImageWidth)

    # Prepare colormap and norms
    cmap = mpl.cm.get_cmap("PiYG")
    norms1 = {
        reactionType: mpn.MidPointNorm(
            midpoint=0,
            vmin=data["min"],
            vmax=data["max"],
        )
        for reactionType, data in maxMinDeltas.items()
    }
    norms2 = findStructureMaxMinDeltas(filteredStructures)

    for mol in filteredStructures.keys():
        imagePath = tempImagesDir + "/molStruct_" + str(mol_id) + ".png"
        imageWidth, imageHeight = PIL.Image.open(imagePath).size
        rowHeight = max(minRowHeight, imageHeight + padding)
        worksheet1.set_row_pixels(
            mol_id + 1,
            rowHeight,
            cell_format=workbook.add_format(
                {
                    "bottom": 3,
                    "align": "center",
                    "valign": "vcenter",
                }
            ),
        )
        worksheet2.set_row_pixels(
            mol_id + 1,
            rowHeight,
            cell_format=workbook.add_format(
                {
                    "bottom": 3,
                    "align": "center",
                    "valign": "vcenter",
                }
            ),
        )
        worksheet3.set_row_pixels(
            mol_id + 1,
            rowHeight,
            cell_format=workbook.add_format(
                {
                    "bottom": 3,
                    "align": "center",
                    "valign": "vcenter",
                }
            ),
        )
        worksheet1.insert_image(
            "A" + str(mol_id + 2),
            imagePath,
            {"x_offset": (maxImageWidth - imageWidth) / 2, "y_offset": int((rowHeight - imageHeight) / 2)},
        )
        worksheet2.insert_image(
            "A" + str(mol_id + 2),
            imagePath,
            {"x_offset": (maxImageWidth - imageWidth) / 2, "y_offset": int((rowHeight - imageHeight) / 2)},
        )
        worksheet3.insert_image(
            "A" + str(mol_id + 2),
            imagePath,
            {"x_offset": (maxImageWidth - imageWidth) / 2, "y_offset": int((rowHeight - imageHeight) / 2)},
        )
        worksheet1.write("B" + str(mol_id + 2), Chem.MolToSmiles(mol))
        worksheet2.write("B" + str(mol_id + 2), Chem.MolToSmiles(mol))
        worksheet3.write("B" + str(mol_id + 2), Chem.MolToSmiles(mol))

        # Non-weighted results
        for reactionType in evaluatedReactionTypes:
            result = filteredStructures[mol][reactionType]
            resultStrings = []
            resultColors1 = []
            resultColors2 = []
            if "pos" in result:
                resultStrings.append(
                    f"+{round(result['pos'].averageDelta, 3)} ({result['pos'].count} occurence{'s' if result['pos'].count > 1 else ''})"
                )
                resultColors1.append(getColorFromDelta(result["pos"].averageDelta, cmap, norms1[reactionType]))
                resultColors2.append(getColorFromDelta(result["pos"].averageDelta, cmap, norms2[mol]))
            if "neg" in result:
                if len(resultStrings) == 0:
                    resultStrings.append(
                        f"{round(result['neg'].averageDelta, 3)} ({result['neg'].count} occurence{'s' if result['neg'].count > 1 else ''})"
                    )
                    resultColors1.append(getColorFromDelta(result["neg"].averageDelta, cmap, norms1[reactionType]))
                    resultColors2.append(getColorFromDelta(result["neg"].averageDelta, cmap, norms2[mol]))
                else:
                    resultStrings.append(
                        f"{round(result['neg'].averageDelta, 3)} ({result['neg'].count} occurence{'s' if result['neg'].count > 1 else ''})"
                    )
                    resultColors1.append(getColorFromDelta(result["neg"].averageDelta, cmap, norms1[reactionType]))
                    resultColors2.append(getColorFromDelta(result["neg"].averageDelta, cmap, norms2[mol]))
            if len(resultStrings) > 0:
                textboxHeight = (rowHeight - 2) / len(resultStrings)
                for i in range(len(resultStrings)):
                    worksheet1.insert_textbox(
                        mol_id + 1,
                        reactionType + 2,
                        resultStrings[i],
                        options={
                            "width": 158,
                            "height": textboxHeight,
                            "x_offset": 1,
                            "y_offset": textboxHeight * i + 1,
                            # TODO: alignment and color
                            "fill": {"color": resultColors1[i]},
                            "align": {"vertical": "middle", "horizontal": "center"},
                        },
                    )
                    worksheet2.insert_textbox(
                        mol_id + 1,
                        reactionType + 2,
                        resultStrings[i],
                        options={
                            "width": 158,
                            "height": textboxHeight,
                            "x_offset": 1,
                            "y_offset": textboxHeight * i + 1,
                            # TODO: alignment and color
                            "fill": {"color": resultColors2[i]},
                            "align": {"vertical": "middle", "horizontal": "center"},
                        },
                    )
            else:
                worksheet1.write(mol_id + 1, reactionType + 2, "/")
                worksheet2.write(mol_id + 1, reactionType + 2, "/")

        # Weighted results
        textboxHeight = rowHeight - 2
        for reactionType in evaluatedReactionTypes:
            result = weightedFilteredStructures[mol][reactionType]  # WeightedStructureDeltaResult
            if result.averageDelta != 0:
                resultColor = getColorFromDelta(result.averageDelta, cmap, weightedStructuresNorms[mol])
                resultString = f"{round(result.averageDelta, 3)} - {result.count} occ{'s' if result.count > 1 else ''} ({round(result.countPercentage * 100, 1)}%)"
                worksheet3.insert_textbox(
                    mol_id + 1,
                    reactionType + 2,
                    resultString,
                    options={
                        "width": 158,
                        "height": textboxHeight,
                        "x_offset": 1,
                        "y_offset": 1,
                        # TODO: alignment and color
                        "fill": {"color": resultColor},
                        "align": {"vertical": "middle", "horizontal": "center"},
                    },
                )
            else:
                worksheet3.write(mol_id + 1, reactionType + 2, "/")

        mol_id += 1

    rowIndex = 0
    for structureId in sortedRows:
        molId = f"mol_{structureId}"
        mol = molIds[molId]
        imagePath = tempImagesDir + "/molStruct_" + str(structureId) + ".png"
        imageWidth, imageHeight = PIL.Image.open(imagePath).size
        rowHeight = max(minRowHeight, imageHeight + padding)
        worksheet4.set_row_pixels(
            rowIndex + 1,
            rowHeight,
            cell_format=workbook.add_format(
                {
                    "bottom": 3,
                    "align": "center",
                    "valign": "vcenter",
                }
            ),
        )
        worksheet4.insert_image(
            "A" + str(rowIndex + 2),
            imagePath,
            {"x_offset": (maxImageWidth - imageWidth) / 2, "y_offset": int((rowHeight - imageHeight) / 2)},
        )
        worksheet4.write("B" + str(rowIndex + 2), Chem.MolToSmiles(mol))
        textboxHeight = rowHeight - 2

        columnIndex = 0
        for reactionType in sortedColumns:
            result = weightedFilteredStructures[mol][reactionType]  # WeightedStructureDeltaResult
            if result.averageDelta != 0:
                resultColor = getColorFromDelta(result.averageDelta, cmap, weightedStructuresNorms[mol])
                resultString = f"{round(result.averageDelta, 3)} - {result.count} occ{'s' if result.count > 1 else ''} ({round(result.countPercentage * 100, 1)}%)"
                worksheet4.insert_textbox(
                    rowIndex + 1,
                    columnIndex + 2,
                    resultString,
                    options={
                        "width": 158,
                        "height": textboxHeight,
                        "x_offset": 1,
                        "y_offset": 1,
                        # TODO: alignment and color
                        "fill": {"color": resultColor},
                        "align": {"vertical": "middle", "horizontal": "center"},
                    },
                )
            else:
                worksheet4.write(rowIndex + 1, columnIndex + 2, "/")

            columnIndex += 1

        rowIndex += 1

    workbook.close()


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
compareReactionsBySubstructures(
    modelPath="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/model/tmpmodel.pt",
    graphsDir="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/data/graphs_with_master_node",
    reactionsDir="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions",
    typesDir="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactionTypes",
    csvDir="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/csv/csvAll",
    molDir="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsComplete",
    tempImagesDir="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/images/structureFinding/comparisonTable",
    outputDir="C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/dgl/visualization/results_structureFinding",
    outputName="structureComparisonTable_test",
    skipIfIncorrectPrediction=True,
    useAbsoluteDeltas=False,
    performCleanEvaluation=False,  # if false function will use saved filtered structures and deltas dictionaries
    evaluatedReactionTypes=list(range(7)),
)
