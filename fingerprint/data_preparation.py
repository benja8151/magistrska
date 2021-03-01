from rdkit import Chem
import os
import pandas as pd

def convertMolToSmiles(mol):
    return Chem.rdmolfiles.MolToSmiles(mol)

def prepareDataset(filetype, input_dir, output_dir):
    directory = os.fsencode(input_dir)
    files = os.listdir(directory)

    filenames_list = []
    smiles_list = []

    for index in range(len(files)):
        try:
            filename = os.fsdecode(files[index])
            if filename.endswith(filetype):

                mol = Chem.MolFromMolFile(input_dir + '/' + filename, removeHs=True)
                
                # Convert mol to RDKit
                smiles = convertMolToSmiles(mol)
                
                if (smiles):
                    filenames_list.append(filename.replace(filetype, ''))
                    smiles_list.append(smiles)
        except:
            pass

        print(str(round((index/len(files) * 100), 2)) + "%")

    series = pd.Series(data=smiles_list, index=filenames_list)
    series.to_csv(output_dir)


prepareDataset(
    '.mol',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsAll',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/smiles/smilesAll.csv'
)