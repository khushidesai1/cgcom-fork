import numpy as np

from scipy import stats
import os
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import networkx as nx



def print_error(value):
    print("error")
    print(value)


def list_files(directory):
    
    originaldictionarys = {}
    for filename in tqdm(os.listdir(directory),desc="readfile"):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            df = pd.read_csv(path,index_col = 0)
            originaldictionarys[filename.split(".")[0]] = df
        # if len(originaldictionarys) >1:
        #     break
    return originaldictionarys
                
def generateSingleCell(args):
    colortypelist =  utils.readcolor()
    # print(args)
    imagefolder, filename,  data_dict,G,nodeidlist,cellidlabel = args
    
    node_colors = ['#D3D3D3' for node in G.nodes()] 
    
    for index, row in data_dict.iterrows():        
        sender, receiver = row['sender'],row['receiver']
        if row["Class"] == "Communication":
            # Add color to node
            senderid = nodeidlist.index(sender)
            receiverid = nodeidlist.index(receiver)            
            node_colors[senderid] = colortypelist[cellidlabel[sender]]  # Color for sender
            node_colors[receiverid] = colortypelist[cellidlabel[receiver]]
            # if not G.has_edge(senderid, receiverid):
            G.add_edge(senderid, receiverid)
       
    plt.figure(figsize=(200, 100))
    pos=nx.get_node_attributes(G,'pos')
    
    nx.draw(G, pos, node_color=node_colors, edge_color='black')
    plt.title('Single Cell communication')
    plt.savefig(imagefolder + filename + '.jpg')
    plt.close()






if __name__=="__main__":
    threshold = 0.01
    rawdirectorypath = "./Script/FeaturelevelGAT/tmp/"
    datasetfolder = "./Dataset/MouseGastrulation/"
    for dataname in ["E1","E2","E3"]:
    # for dataname in ["E1"]:
        print(dataname)
        Wholefilepath = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_hightlight_pvalue_csv/"
        singlecelllabelfilepath = datasetfolder+dataname+"_label.csv"
        cellidlabel = utils.getcelllabel(singlecelllabelfilepath,sep = ",")
        data_dicts = list_files(Wholefilepath)
        
        locationfilepath = datasetfolder+dataname+"_location.csv"
        G,disdict,locationlist,nodeidlist,minlocation,maxlocation= utils.buildgraph(locationfilepath,sep=",",title=True)
        G_directed = nx.DiGraph(G)
        imagefolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_singlecell_communication_pvalue_image/"
        os.makedirs(imagefolder, exist_ok=True)


        with Pool(20) as p:
            tasks = []
            for filename,data_dict in data_dicts.items():   
                task = (imagefolder, filename,  data_dict,G_directed,nodeidlist,cellidlabel)
                tasks.append(task)  
            
            pbar = tqdm(total=len(tasks),desc="Building single cell map")
            def update(*a):
                pbar.update()

            for task in tasks:                
                p.apply_async(generateSingleCell, (task,),error_callback=print_error,callback=update)

            p.close()
            p.join()








            




    
    











