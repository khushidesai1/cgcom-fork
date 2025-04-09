
import pandas as pd
import utils
import pickle
from tqdm import tqdm
if __name__=="__main__":

       

    datasetfolder = "./Dataset/MouseGastrulation/"
    dataname = "E1"
    # singlecelldf = utils.loadscfile(datasetfolder+"Version1_total_100/"+dataname+"_sub_expression.csv")
    singlecelldf = utils.loadscfile(datasetfolder+dataname+"_expression.csv")
    
    LRfilepath = "./Knowledge/allr.csv"
    lrmapping = utils.load_csv_and_create_dict(LRfilepath)
    # subLRdict = utils.generate_sub_dictionary(lrmapping, list(singlecelldf.columns))
    subLRdict =utils.pick_random_keys_with_elements(lrmapping,list(singlecelldf.columns), n=25, max_elements=2)
    TFfilepath = "./Knowledge/TFlist.txt"
    TFs = utils.loadtf(TFfilepath)
    # selectedTFs = utils.pick_random_common_elements(list(singlecelldf.columns), TFs)
    selectedTFs = utils.pick_random_common_elements(list(singlecelldf.columns), TFs, n=25)
    # print(subLRdict)
    ligands = []
    # lrdictionary = {}
    selectedgenes = []
    for key, value in subLRdict.items():
        selectedgenes= value + selectedgenes
        selectedgenes.append(key)
        ligands+=value

    mask_indexes = utils.generateMaskindex(ligands,subLRdict)
    selectedgenes+=selectedTFs
    subdf = utils.formsubdataframe(singlecelldf, selectedgenes)
    # print(subdf.columns)
    singlecelllabelfilepath = datasetfolder+dataname+"_label.csv"
    cellidlabel = utils.getcelllabel(singlecelllabelfilepath,sep = ",")

    locationfilepath = datasetfolder+dataname+"_location.csv"
    G,disdict,locationlist,nodeidlist,minlocation,maxlocation= utils.buildgraph(locationfilepath,sep=",",title=True)
    neighborthresholdratio = 0.02
    G,edgelist = utils.readdedgestoGraph(G,locationlist,disdict,neighborthresholdratio,minlocation,maxlocation)


    orgiganlnodeids=[]
    features=[]
    edges = []
    labels = []

    subgraphs = utils.generate_subgraphs(G)
    for subgraphnode,subg in tqdm(subgraphs.items(),total=len(subgraphs),desc="Creating subgraph"):
        # print(subgraphnode)
        # print(subg.nodes())
        # print(subg.edges())
        # print(utils.get_adjacency_matrix(subg))

        edgelistsource = []
        edgelisttarget = []
        
        featurelest = []
        label = cellidlabel[nodeidlist[subgraphnode]]

        index = 0
        for node in subg.nodes():
            mainfeature = subdf.loc[nodeidlist[node]].tolist()
            featurelest.append(mainfeature)
            if index>0:
                edgelistsource.append(0)
                edgelisttarget.append(index)
            index+=1

        orgiganlnodeids.append(subg.nodes())
        features.append(featurelest)
        labels.append(label)
        edges.append([edgelistsource,edgelisttarget])

    print(len(selectedgenes))
    
    # print(features[0])
    print(len(features))
    # print(features[0][0])
    print(len(features[0][0]))

    pikcleoutputfile =  "./Script/FeaturelevelGAT/tmp/test.pkl"
    with open(pikcleoutputfile, 'wb') as f:  
        pickle.dump([G,edges,features, labels,orgiganlnodeids,nodeidlist,cellidlabel,mask_indexes,ligands,subLRdict,selectedTFs], f)





