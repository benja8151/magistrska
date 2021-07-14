import dgl
import pandas as pd
import torch
import numpy as np
import itertools
import sys
import pickle
import os

def filterReactionsConnections(csvPath: str, connections: np.ndarray, node_indexes):
    #original_length = len(connections)
    compounds = {}
    filteredConnections = []
    for pair in connections:
        cmp1 = node_indexes[pair[0]]['cmp']
        if (cmp1 not in compounds):
            compounds[cmp1] = pd.read_csv(csvPath + '/' + cmp1 + '/nodes.csv')
        cmp2 = node_indexes[pair[1]]['cmp']
        if (cmp2 not in compounds):
            compounds[cmp2] = pd.read_csv(csvPath + '/' + cmp2 + '/nodes.csv')
        if (compounds[cmp1]['atomic_number'][node_indexes[pair[0]]['ind']] == compounds[cmp2]['atomic_number'][node_indexes[pair[1]]['ind']]):
            filteredConnections.append(pair)

    #new_length = len(filteredConnections)
    #print("Reduced reactions connections by: ", round(((original_length-new_length)/original_length * 100), 2), "%")

    return np.array(filteredConnections).astype(connections.dtype)

def reactionToGraph(reactionsPath: str, reactionName: str, csvPath: str):
    try:
        reactionFile = open(reactionsPath + '/' + reactionName)
        reactionSides = reactionFile.read().split('-')
        reactionFile.close()

        leftSide = reactionSides[0].split(',')
        rightSide = reactionSides[1].split(',')

        graphData = {
            'reacts': (np.array([], dtype=int), np.array([], dtype=int)),
            'inter': (np.array([], dtype=int), np.array([], dtype=int)),
            'intra': (np.array([], dtype=int), np.array([], dtype=int)),
            'master': (np.array([], dtype=int), np.array([], dtype=int)),
        }
        nodeCount = 0

        leftSideNodes = []
        rightSideNodes = []
        node_counts = {}

        # Reactants
        for compound in leftSide:
            nodes = pd.read_csv(csvPath + '/' + compound + '/nodes.csv')
            edges = pd.read_csv(csvPath + '/' + compound + '/edges.csv')
            
            srcEdges = edges['src'].to_numpy().astype(int) + nodeCount
            dstEdges = edges['dst'].to_numpy().astype(int) + nodeCount

            # Make bi-directional edges
            u = np.concatenate([srcEdges, dstEdges])
            v = np.concatenate([dstEdges, srcEdges])

            graphData['intra'] = (np.append(graphData['intra'][0], u), np.append(graphData['intra'][1], v)) 

            leftSideNodes.append(np.arange(nodeCount, nodeCount + len(nodes), dtype=int))

            node_counts[compound] = len(nodes)

            nodeCount += len(nodes)

        # Products
        for compound in rightSide:
            nodes = pd.read_csv(csvPath + '/' + compound + '/nodes.csv')
            edges = pd.read_csv(csvPath + '/' + compound + '/edges.csv')

            srcEdges = edges['src'].to_numpy().astype(int) + nodeCount
            dstEdges = edges['dst'].to_numpy().astype(int) + nodeCount

            # Make bi-directional edges
            u = np.concatenate([srcEdges, dstEdges])
            v = np.concatenate([dstEdges, srcEdges])

            graphData['intra'] = (np.append(graphData['intra'][0], u), np.append(graphData['intra'][1], v)) 

            rightSideNodes.append(np.arange(nodeCount, nodeCount + len(nodes), dtype=int))

            node_counts[compound] = len(nodes)

            nodeCount += len(nodes)

        # Master Nodes
        currentNode = 0
        compoundIndex = 0
        leftSideMasterNodes = list(range(nodeCount, nodeCount + len(leftSideNodes)))
        rightSideMasterNodes = list(range(nodeCount + len(leftSideNodes), nodeCount + len(leftSideNodes) + len(rightSideNodes)))
        for compound in leftSide + rightSide:
            src = np.full(node_counts[compound], nodeCount + compoundIndex, dtype=int)
            dst = np.arange(currentNode, node_counts[compound] + currentNode, dtype=int)
            currentNode += node_counts[compound]
            compoundIndex += 1
            u = np.concatenate([src, dst])
            v = np.concatenate([dst, src])
            graphData['master'] = (np.append(graphData['master'][0], u), np.append(graphData['master'][1], v))
        
        # Inter-connections on each side of reaction
        if (len(leftSideNodes) > 1):
            combinations = list(itertools.combinations(leftSideMasterNodes, 2))
            for combination in combinations:
                graphData['inter'] = (
                    np.append(graphData['inter'][0], np.concatenate([[combination[0], combination[1]]])), 
                    np.append(graphData['inter'][1], np.concatenate([[combination[1], combination[0]]])),
                ) 
        if (len(rightSideNodes) > 1):
            combinations = list(itertools.combinations(rightSideMasterNodes, 2))
            for combination in combinations:
                graphData['inter'] = (
                    np.append(graphData['inter'][0], np.concatenate([[combination[0], combination[1]]])), 
                    np.append(graphData['inter'][1], np.concatenate([[combination[1], combination[0]]])),
                ) 
        
        # Reaction connections
        connections = np.array(np.meshgrid(leftSideMasterNodes, rightSideMasterNodes)).T.reshape(-1,2)
        graphData['reacts'] = (
            np.concatenate([connections[:,0], connections[:,1]],), 
            np.concatenate([connections[:,1], connections[:,0]]),
        )

        g = dgl.graph((
            np.concatenate([graphData['intra'][0], graphData['master'][0], graphData['inter'][0], graphData['reacts'][0]]),
            np.concatenate([graphData['intra'][1], graphData['master'][1], graphData['inter'][1], graphData['reacts'][1]])
        ))

        # Atom and bond features
        atomic_number = np.array([])
        valence = np.array([])
        charge = np.array([])
        degree = np.array([])
        hydrogens = np.array([])
        radicals = np.array([])
        aromacity = np.array([])
        #mass = np.array([])
        #vdw = np.array([])
        btype = np.array([])
        baromacity = np.array([])
        conjugation = np.array([])

        for compound in leftSide + rightSide:
            nodes = pd.read_csv(csvPath + '/' + compound + '/nodes.csv')
            edges = pd.read_csv(csvPath + '/' + compound + '/edges.csv')

            atomic_number = np.concatenate([atomic_number, nodes['atomic_number'].to_numpy()])
            valence = np.concatenate([valence, nodes['valence'].to_numpy()])
            charge = np.concatenate([charge, nodes['charge'].to_numpy()])
            degree = np.concatenate([degree, nodes['degree'].to_numpy()])
            hydrogens = np.concatenate([hydrogens, nodes['hydrogens'].to_numpy()])
            radicals = np.concatenate([radicals, nodes['radicals'].to_numpy()])
            aromacity = np.concatenate([aromacity, nodes['aromacity'].to_numpy()])
            # temporarily removed
            #mass = np.concatenate([mass, nodes['mass'].to_numpy()])
            #vdw = np.concatenate([vdw, nodes['VdW_radius'].to_numpy()])

            btype = np.concatenate([btype, edges['type'].to_numpy()])
            baromacity = np.concatenate([baromacity, edges['aromacity'].to_numpy()])
            conjugation = np.concatenate([conjugation, edges['conjugation'].to_numpy()])
            
            # reversed src and dst
            btype = np.concatenate([btype, edges['type'].to_numpy()])
            baromacity = np.concatenate([baromacity, edges['aromacity'].to_numpy()])
            conjugation = np.concatenate([conjugation, edges['conjugation'].to_numpy()])

        node_feats = np.column_stack((
            atomic_number,
            valence,
            charge + 5, # +5 to remove negative numbers
            degree,
            hydrogens,
            radicals,
            aromacity,
            #mass,
            #vdw
        )).astype(float)
        edge_feats = np.column_stack((
            btype + 3,
            #baromacity,
            #conjugation
        )).astype(float)

        node_feats = np.concatenate([node_feats, np.full((len(leftSideMasterNodes + rightSideMasterNodes), node_feats.shape[1]), 0, dtype=float)])
        edge_feats = np.concatenate([edge_feats, np.full((len(graphData['master'][0]), 1), 0, dtype=float)])
        edge_feats = np.concatenate([edge_feats, np.full((len(graphData['inter'][0]), 1), 1, dtype=float)])
        edge_feats = np.concatenate([edge_feats, np.full((len(graphData['reacts'][0]), 1), 2, dtype=float)])
        
        #print("-----------")
        #print(node_feats.shape)
        #print(edge_feats.shape)
        #print(g.num_nodes())
        #print(g.num_edges())
        #print(btype)
        g.ndata['feat'] = torch.tensor(node_feats).int()
        g.edata['feat'] = torch.tensor(edge_feats).int()
        
        return g

    except:
        print("Error in reaction: " + reactionName)
        print(sys.exc_info())
        return None

def saveGraph(g, saveDir: str, saveName: str):
    with open(saveDir + '/' + saveName, 'wb') as file:
        pickle.dump(g, file)
    file.close()

def loadGraph(saveDir: str, saveName: str):
    with open(saveDir + '/' + saveName, 'rb') as file:
        g = pickle.load(file)
    file.close()
    return g

def createGraphs(reactionsDir: str, outputDir: str, csvPath: str):
    _, _, all_filenames = next(os.walk(reactionsDir))

    count = 0
    for reaction in all_filenames:
        g = reactionToGraph(reactionsDir, reaction, csvPath)
        if (g != None):
            saveGraph(g, outputDir, reaction)
        
        print(str(round((count/len(all_filenames) * 100), 2)) + "%")
        count +=1
#############################################################################

createGraphs(
    '/home/bsmrdelj/local/git/magistrska/data/reactions/EnzymaticReactions',
    '/home/bsmrdelj/local/git/magistrska/dgl/data/graphs_with_master_node',
    '/home/bsmrdelj/local/git/magistrska/data/csv/csvAll',
)

#reactionToGraph(
#    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions',
#    'R00912',
#    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/csv/csvAll'
#)

