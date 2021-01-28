from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import os
import csv

def mapFiles(directory_string: str, output_directory: str, filetype: str) -> None:
    directory = os.fsencode(directory_string)
    files = os.listdir(directory)

    for index in range(len(files)):
        filename = os.fsdecode(files[index])
        if filename.endswith(filetype):
            
            # Convert mol to RDKit
            m = molToRDKit(directory_string + '/' + filename)
            
            if (m):
                # Format .csv
                nodes, edges, properties = RDKitToCSV(m, index)

                # Save data
                filename = filename.replace(filetype, '')

                try: 
                    os.makedirs(output_directory + "/" + filename)
                except:
                    # Directory already exists
                    pass

                createCSV(output_directory + "/" + filename, 'nodes', nodes)
                createCSV(output_directory + "/" + filename, 'edges', edges)
                createCSV(output_directory + "/" + filename, 'properties', properties)


        print(str(round((index/len(files) * 100), 2)) + "%")

def molToRDKit(filepath: str):
    try:
        m = Chem.MolFromMolFile(filepath, removeHs=False)
        m = Chem.AddHs(m)
        Chem.AssignAtomChiralTagsFromStructure(m)
        
        # TODO: skipping for now, error with some molecules
        #AllChem.EmbedMolecule(m,useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
        #AllChem.MMFFOptimizeMolecule(m)
        
        return m
    except:
        print(filepath + " is not a valid .mol file")
        return False

def createCSV(directory_string: str, filename: str, data):
    """Writes csv data"""

    with open(directory_string + '/' + filename + '.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for row in data:
            writer.writerow(row)

def RDKitToCSV(m, index):
    """Formats RDKit molecule to .csv data."""
    
    #Nodes
    nodes = [['atomic_number', 'valence', 'charge', 'degree', 'hydrogens', 'radicals', 'aromacity', 'mass', 'VdW_radius']]
    for atom in m.GetAtoms():
        nodes.append(getAtomFeatures(atom))

    #Edges
    edges = [['src', 'dst', 'type', 'aromacity', 'conjugation']]
    for bond in m.GetBonds():
        edges.append(getBondFeatures(bond))
        
    #Properties
    properties = [['weight', 'alogp', 'tpsa', 'hbd'], getMoleculeProperties(m, index)]

    return nodes, edges, properties

def getAtomFeatures(a: Chem.rdchem.Atom):
    return [
        # a.GetIdx(),
        a.GetAtomicNum(),
        a.GetTotalValence(),
        a.GetFormalCharge(),
        a.GetDegree(),
        a.GetTotalDegree() - a.GetDegree(),
        a.GetNumRadicalElectrons(),
        int(a.GetIsAromatic()),
        a.GetMass(),
        Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum())
    ]

def getBondFeatures(b: Chem.rdchem.Bond):
    return [
        b.GetBeginAtomIdx(),
        b.GetEndAtomIdx(),
        b.GetBondTypeAsDouble(),
        int(b.GetIsAromatic()),
        int(b.GetIsConjugated())
    ]

def getMoleculeProperties(m: Chem.rdchem.Mol, index: int):
    qed = QED.properties(m)
    return [
        #index,
        Descriptors.ExactMolWt(m),
        qed[1],
        qed[4],
        qed[3]
    ]

def formatMols(inputDirectory: str, outputDirectory: str):
    mapFiles(
        inputDirectory, 
        outputDirectory,
        '.mol'
     )

######################################################################

formatMols(
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsAll',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/csv/csvAll'
)
