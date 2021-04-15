import requests
import os

# Utils
def saveFile(data, filename, filedir):
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    f = open(f"{filedir}/{filename}", "w")
    f.write(data)
    f.close()


# Compounds
def getCompounds(outputDir):
    url = "http://rest.kegg.jp/list/compound"

    response = requests.get(url)

    if (response.status_code == 200):
        body = response.text
        compounds = body.splitlines()
        
        n_compounds = len(compounds)
        for i in range(n_compounds):
            compoundId = compounds[i][4 : 10]
            mol = getMol(compoundId)
            if (mol is not None):
                saveFile(mol, f"{compoundId}.mol", outputDir)
            print(i/n_compounds * 100, "%")
    else:
        raise Exception("Network Error")

def getMol(compoundId):
    url = f"http://rest.kegg.jp/get/{compoundId}/mol"

    response = requests.get(url)

    if (response.status_code == 200):
        return response.text
    else:
        print(f"Can't retrieve mol file: {compoundId}")
        return None


# Reactions
def getAllReactions():
    url = "http://rest.kegg.jp/list/reaction"

    response = requests.get(url)

    if (response.status_code == 200):
        body = response.text
        reactions = body.splitlines()
        print(len(reactions))
        # TODO
    else:
        raise Exception("Network Error")

def getEnzymaticReactionTypes(outputDir):
    url = "http://rest.kegg.jp/get/br:br08201/json"
    response = requests.get(url)

    allReactionsNames = set()

    if (response.status_code == 200):
        body = response.json()
        
        for reactionType in body["children"]:
            formattedReactionType = reactionType["name"].split(' ')[1]
            formattedReactionNames = ""
            
            for second_class in reactionType["children"]:
                for third_class in second_class["children"]:
                    for fourth_class in third_class["children"]:
                        try:
                            for reaction in fourth_class["children"]:
                                try:
                                    reactionName = reaction["name"].split(' ')[0]
                                    if (reactionName in allReactionsNames):
                                        continue
                                        #print("Non-Unique reaction: " + reactionName)
                                    else:
                                        allReactionsNames.add(reactionName)
                                        formattedReactionNames += f"{reactionName}\n"
                                except:
                                    continue
                        except:
                            continue
            saveFile(formattedReactionNames, formattedReactionType, outputDir)

    else:
        raise Exception("Network Error")

def getReaction(reactionId):
    url = f"http://rest.kegg.jp/get/{reactionId}"

    response = requests.get(url)

    if (response.status_code == 200):
        body = response.text.splitlines()
        equation = None
        for line in body:
            if line.startswith("EQUATION"):
                equation = line
                break

        if (equation is not None):
            reaction_sides = equation.split("<=>")
            left_side = ",".join(list(filter(lambda x: x.startswith("C"), reaction_sides[0].split(' '))))
            right_side = ",".join(list(filter(lambda x: x.startswith("C"), reaction_sides[1].split(' '))))
            
            return f'{left_side}-{right_side}'

        else:
            print(f"Can't find equation: {reactionId}")
            return None
    else:
        print(f"Can't retrieve reaction file: {reactionId}")
        return None

def getEnzymaticReactions(inputDir, outputDir):
    total_reactions = 11505
    i=0
    for subdir, _, files in os.walk(inputDir):
        for file in files:
            f = open(os.path.join(subdir, file), 'r')
            for reactionId in f:
                reactionId = reactionId.rstrip()
                reaction = getReaction(reactionId)
                saveFile(reaction, reactionId, outputDir)
                i+=1
                print(100 * i/total_reactions, "%")

def mapRBPs(rbpListInput, outputDir):
    url = "http://rest.kegg.jp/list/orthology"
    response = requests.get(url)

    rbpList = open(rbpListInput, 'r')
    rbpSet = set()
    for rbp in rbpList:
        rbpSet.add(rbp.rstrip().lower())
        
    if (response.status_code == 200):
        body = response.text.splitlines()
        kegg_list = ""

        totalCount = len(body)
        foundCount = 0
        checkedCount = 0

        for line in body:
            line_split = line.split('\t')
            kegg_code = line_split[0][3:]
            kegg_names = line_split[1].split(';')[0].split(', ')
            
            foundMatch = False

            for kegg_name, rbp_name in ((x, y) for x in kegg_names for y in rbpSet):
                if kegg_name.lower() in rbp_name:
                    foundMatch = True
                    break
            
            if (foundMatch):
                foundCount +=1
                kegg_list += f"{kegg_code}\n"

            checkedCount +=1
            print(100 * checkedCount/totalCount, "%")
        
        saveFile(kegg_list, "rbp_list_kegg", outputDir)

        print("Total Count: ", totalCount)
        print("Found Count: ", foundCount)

    else:
        raise Exception("Network Error")

def getReactionsWithRBPs(rbpListInput, outputDir):
    rbpList = open(rbpListInput, 'r')
    reactions = set()

    rbp_count = 12635
    count = 0

    for rbp in rbpList:
        url = f"http://rest.kegg.jp/link/reaction/{rbp}"
        response = requests.get(url)

        if (response.status_code == 200):
            for line in response.text.splitlines():
                try:
                    reaction = line.split(':')[2]
                    reactions.add(reaction)
                except:
                    pass
        else:
            print("Network Error with: ", rbp)

        count += 1
        print(100 * count/rbp_count, "%")
    
    output = ""
    for reaction in reactions:
        output += f"{reaction}\n"
    
    saveFile(output.rstrip(), "reactions_RBP", outputDir)
    print("Reaction count: ", len(reactions))
    
#####################################################################

#getCompounds('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/mols/MolsComplete')

#getAllReactions()
#getEnzymaticReactionTypes('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactionTypes')
#getEnzymaticReactions('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactionTypes', 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions/EnzymaticReactions')
#mapRBPs('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/enzymes/rbp_list', 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/enzymes')
getReactionsWithRBPs('C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/enzymes/rbp_list_kegg', 'C:/Users/Benjamin/Documents/Datoteke_za_solo/MAG/magistrska/data/reactions')