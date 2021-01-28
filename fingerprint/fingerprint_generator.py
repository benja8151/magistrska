import os 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch_geometric
import os
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree
import csv
from torch_geometric.data.dataloader import DataLoader
import numpy as np

class NeuralLoop(MessagePassing):
    def __init__(self, atom_features, fp_size):
        super(NeuralLoop, self).__init__(aggr='add')
        self.H = nn.Linear(atom_features, atom_features)
        self.W = nn.Linear(atom_features, fp_size)
        
    def forward(self, x, edge_index):
        # x shape: [Number of atoms in molecule, Number of atom features]; [N, in_channels]
        # edge_index shape: [2, E]; E is the number of edges
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_j, edge_index, size):
        # We simply sum all the neighbouring nodes (including self-loops)
        # This is done implicitly by PyTorch-Geometric :)
        return x_j 
    
    def update(self, v):
        
        updated_atom_features = self.H(v).sigmoid()
        updated_fingerprint = self.W(updated_atom_features).softmax(dim=-1)
        
        return updated_atom_features, updated_fingerprint # shape [N, atom_features]
    

class NeuralFP(nn.Module):
    def __init__(self, atom_features=52, fp_size=2048):
        super(NeuralFP, self).__init__()
        
        self.atom_features = atom_features
        self.fp_size = fp_size
        
        self.loop1 = NeuralLoop(atom_features=atom_features, fp_size=fp_size)
        self.loop2 = NeuralLoop(atom_features=atom_features, fp_size=fp_size)
        self.loops = nn.ModuleList([self.loop1, self.loop2])
        
    def forward(self, data):
        fingerprint = torch.zeros((data.batch.shape[0], self.fp_size), dtype=torch.float)
        
        out = data.x
        print(type(data.edge_index))
        
        for idx, loop in enumerate(self.loops):
            updated_atom_features, updated_fingerprint = loop(out, data.edge_index)
            out = updated_atom_features
            fingerprint += updated_fingerprint
            
        return scatter_add(fingerprint, data.batch, dim=0)


def open_csv_files(root_dir: str):
    data_list = []
    
    for subdir, dirs, files in os.walk(root_dir):
        try:
            nodes = get_nodes((os.path.join(subdir, 'nodes.csv')))
            edges = get_edges((os.path.join(subdir, 'edges.csv')))
            data = torch_geometric.data.data.Data(x=nodes, edge_index=edges)
            data_list.append(data)
        except:
            pass
        
    return DataLoader(data_list, batch_size = 1, shuffle = False), data_list
            

def get_nodes(file: str):
    output = []
    
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            output.append(np.asarray(row, dtype=np.float32))
    
    return torch.tensor(output, dtype = torch.float32)


def get_edges(file: str):
    xs, ys = [], []
    
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        
        
        for row in reader:
            src = row[0]
            dst = row[1]
            xs += [src, dst]
            ys += [dst, src]
    
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    
    return torch.tensor([xs, ys], dtype = torch.long)


def generate_fps(root_dir: str, output_dir: str):
    dloader, dlist = open_csv_files(root_dir)
    neural_fp = NeuralFP(atom_features = 9, fp_size = 2048)
    
    for step, batch in enumerate(dloader):
        fp = neural_fp(batch)
        filename = str(step)
        torch.save(fp, output_dir + '/' + filename + '.pt')



#######################################################################################################

generate_fps(
    "C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/csv/csvAll", 
    "C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/fingerprint/generated"
)