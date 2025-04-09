
# Version 1 we will only have 100 gene.
# 50 ligands, 25 receptors and 25 TFs


import pandas as pd
import utils

if __name__=="__main__":
    
    # load dataset
    datasetfolder = "./Dataset/MouseGastrulation/"
    dataname = "E1"


    singlecelldf = utils.loadscfile(datasetfolder+dataname+"_expression.csv")
    # print(singlecelldf.index)
    TFfilepath = "./Knowledge/TFlist.txt"
    TFs = utils.loadtf(TFfilepath)
    selectedTFs = utils.pick_random_common_elements(list(singlecelldf.columns), TFs, n=25)
    print(selectedTFs)
    # randomely get and receptor
    LRfilepath = "./Knowledge/allr.csv"
    lrmapping = utils.load_csv_and_create_dict(LRfilepath)
    # sub_LR = utils.generate_sub_dictionary(lrmapping, list(singlecelldf.columns))
    selected_LR = utils.pick_random_keys_with_elements(lrmapping, list(singlecelldf.columns), n=25, max_elements=2)
    print(selected_LR)

    ligands = []
    # lrdictionary = {}
    selectedgenes = []
    for key, value in selected_LR.items():
        selectedgenes= value + selectedgenes
        selectedgenes.append(key)
        ligands+=value
    
    selectedgenes+=selectedTFs

    # subdf = utils.formsubdataframe(singlecelldf, selectedgenes)
    # print(subdf.columns)
    # subdf.to_csv(datasetfolder+"Version1_total_100/"+dataname+"_sub_expression.csv")
    # p1_channels = len(ligands)
    # p2_channels = len(selected_LR)
    # print(utils.generateMask(ligands,selected_LR,p1_channels,p2_channels))