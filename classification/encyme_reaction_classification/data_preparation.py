import os
import pandas as pd

reactions_dir = 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions'

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

def createReactionsCSV(all_dir, rbp_file,types_dir, output_dir):
    _, _, all_filenames = next(os.walk(all_dir))
    _, _, reaction_types = next(os.walk(types_dir))
    
    reactions_2_2 = []
    hasRBP = []
    types = []

    rbps = open(rbp_file, 'r')
    rbps_list = rbps.read().splitlines()
    
    types_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
    }
    for reaction_type in reaction_types:
        type_int = getReactionType(reaction_type)
        types_dict[type_int] = open(types_dir + "/" + reaction_type, 'r').read().splitlines()

    def getType(reactionId):
        if (reactionId in types_dict[0]):
            return 0
        if (reactionId in types_dict[1]):
            return 1
        if (reactionId in types_dict[2]):
            return 2
        if (reactionId in types_dict[3]):
            return 3
        if (reactionId in types_dict[4]):
            return 4
        if (reactionId in types_dict[5]):
            return 5
        if (reactionId in types_dict[6]):
            return 6
        if (reactionId in types_dict[7]):
            return 7

    for reaction in all_filenames:
        f = open(reactions_dir + "/" + reaction)
        sides = f.read().split('-')
        # Filter only 2 -> 2 reactions
        if (len(sides[0].split(",")) == 2 and len(sides[1].split(",")) == 2):
            reactions_2_2.append(reaction)
            if (reaction in rbps_list):
                hasRBP.append(1)
            else:
                hasRBP.append(0)
            types.append(getType(reaction))

        f.close()
    
    df = pd.DataFrame(data={'Reaction': reactions_2_2, 'hasRBP': hasRBP, "reactionType": types})
    df.to_csv(output_dir, index=False)

    rbps.close()

createReactionsCSV(
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/reactions_RBP',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactionTypes',
    'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/classification/encyme_reaction_classification/data/reactions.csv'
)