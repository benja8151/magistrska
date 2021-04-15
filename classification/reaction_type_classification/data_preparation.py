import os
import pandas as pd

def getReactionType(name):
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

    if (name == "Hydrolase"):
        return 0
    elif (name == "Isomerase"):
        return 1
    elif (name == "Ligase"):
        return 2
    elif (name == "Lyase"):
        return 3
    elif (name == "Oxidoreductase"):
        return 4
    elif (name == "Transferase"):
        return 5
    elif (name == "Translocase"):
        return 6
    elif (name == "Unassigned"):
        return 7
    else:
        raise Exception("Unsupported reaction type: " + name)

def createReactionsCSV(all_dir, types_dir, output_dir):
    _, _, all_filenames = next(os.walk(all_dir))
    _, _, reaction_types = next(os.walk(types_dir))

    reactions = []
    types = []

    type_count = {}

    for reaction_type in reaction_types:
        type_int = getReactionType(reaction_type)
        f = open(types_dir + "/" + reaction_type, 'r')
        for reaction in f:
            f_reaction = open(all_dir + "/" + reaction.strip())
            sides = f_reaction.read().split('-')

            # Filter only 2 -> 2 reactions
            if (len(sides[0].split(",")) == 2 and len(sides[1].split(",")) == 2):
                if (reaction.strip() in all_filenames):
                    reactions.append(reaction.strip())
                    types.append(type_int)
                    if (reaction_type in type_count):
                        type_count[reaction_type] +=1
                    else:
                        type_count[reaction_type] = 1

            f_reaction.close()
        f.close()
    
    df = pd.DataFrame(data={'Reaction': reactions, 'Type': types}).sample(frac=1)
    df.to_csv(output_dir, index=False)

    print("Reaction count:")
    print(type_count)

createReactionsCSV(
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactionTypes',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/reaction_type_classification/data/reactions.csv'
)