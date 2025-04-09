from torch_geometric.data import DataLoader
from torch.optim import Adam
import torch

import torch_geometric.utils 
import random
import torch.nn.functional as F
from model import GATGraphClassifier
import pickle
import utils
import pandas as pd
import utils
import pickle
from tqdm import tqdm
import wandb
from torch.utils.data import  Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle
from collections import Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from multiprocessing import Pool
import multiprocessing
import gc

gc.collect()
dataname = "E1"
neighborthresholdratio = 0.01

# singlecelllabelfilepath = datasetfolder+dataname+"_label.csv"
# cellidlabel = utils.getcelllabel(singlecelllabelfilepath,sep = ",")


outputpath = "./Script/FeaturelevelGAT/tmp/"+dataname+"_"+str(neighborthresholdratio)+"/"

pikcleoutputfile = outputpath+"Feature_"+dataname+".pkl"
with open(pikcleoutputfile, 'rb') as f:  
    ligands,receptors, selectedTFs,subLRdict = pickle.load(f)

alllrdict={}
writenodelist =[]

for receptor,subLRligands in subLRdict.items():
    for ligand in subLRligands:
        if not ligand+"-"+receptor in alllrdict:
            alllrdict[ligand+"-"+receptor]=[]

# outputpath+'nodelist-'+str(list(nodelists)[0])+'.pkl'
for i in tqdm(range(0,21000)):
    if os.path.exists(outputpath+'firstattention-'+str(i)+".pt") and os.path.exists(outputpath+'nodelist-'+str(i)+'.pkl')and os.path.exists(outputpath+'allnodeid-'+str(i)+'.pkl'):
        firstattention = torch.load(outputpath+'firstattention-'+str(i)+".pt")
        # print(firstattention.shape)
        with open(outputpath+'nodelist-'+str(i)+'.pkl', 'rb') as f:  
            nodelists = pickle.load(f)            
        with open(outputpath+'allnodeid-'+str(i)+'.pkl', 'rb') as f:  
            allnodeid = pickle.load(f)
        
        totaldom = 1/firstattention.size(0)
        # print(totaldom)
        senderid = allnodeid[0]    
        for j in range(firstattention.size(0)):
            receptorid = allnodeid[j]
            matrix_df = pd.DataFrame(firstattention[j,:,:].numpy(), index=receptors, columns=ligands)
            # matrix_df.to_csv("./testdf.csv",index=True)
            # print(matrix_df)
            # exit()
            if not senderid+"-"+receptorid in writenodelist:
                writenodelist.append(senderid+"-"+receptorid)
            for receptor,subLRligands in subLRdict.items():
                for ligand in subLRligands:
                    # print(matrix_df.loc[receptor, ligand])
                    value = matrix_df.loc[receptor, ligand] / totaldom
                    # print(value)
                    alllrdict[ligand+"-"+receptor].append(value)

resultoutputpath = "./Script/FeaturelevelGAT/tmp/"+dataname+"_"+str(neighborthresholdratio)+"_results/"
if not os.path.exists(resultoutputpath):
    os.makedirs(resultoutputpath)
for lrpair,values in alllrdict.items():
    with open(resultoutputpath+lrpair+".txt", "w") as f:
        f.write("SR\tCS\n")
        for name,CS in zip(writenodelist,values):
            f.write(name+"\t"+str(CS)+"\n")




































