import os
from numpy.core.numeric import False_
import pandas as pd

def createReactionsCSV(all_dir, hsa_dir, output_dir):
    _, _, all_filenames = next(os.walk(all_dir))
    _, _, hsa_filenames = next(os.walk(hsa_dir))
    
    isHsa = []

    for reaction in all_filenames:
        if (reaction in hsa_filenames):
            isHsa.append(1)
        else:
            isHsa.append(0)
    
    df = pd.DataFrame(data={'Reaction': all_filenames, 'isHsa': isHsa})
    df.to_csv(output_dir, index=False)

createReactionsCSV(
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/ParsedAll',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/ParsedHsa',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/data/reactions.csv'
)