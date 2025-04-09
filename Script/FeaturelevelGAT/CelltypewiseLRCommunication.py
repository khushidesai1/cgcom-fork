import numpy as np

from scipy import stats
import os
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from collections import defaultdict

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
        
    return originaldictionarys

def generateheatmap(args):
    csvfolder, data_dicts, sendertype, receivertype = args
    # print("start", sendertype, receivertype)
    
    # Use defaultdict to initialize missing keys with 0
    resultsdataframe = defaultdict(int)
    
    for filename, data_dict in data_dicts.items():
        # Filter rows where Class is Communication and sender/receiver types match
        filtered_data = data_dict[(data_dict["Class"] == "Communication") & 
                                  (data_dict["sender"].map(cellidlabel) == sendertype) & 
                                  (data_dict["receiver"].map(cellidlabel) == receivertype)]
        
        # Sum CCI_score for each file
        resultsdataframe[filename] += filtered_data["CCI_score"].sum()
    
    # Convert results to DataFrame for saving to CSV
    df = pd.DataFrame(list(resultsdataframe.items()), columns=['LRpairs', 'Total_CCI'])
    df.to_csv(f"{csvfolder}{sendertype}_{receivertype}.csv", index=False)
    
    # print("Done", sendertype, receivertype)

# def generateheatmap(args):   
    
#     csvfolder, data_dicts,sendertype,receivertype = args   
#     print("start",sendertype,receivertype) 
#     resultsdataframe={}
#     for filename,data_dict in data_dicts.items():
#         if not filename in resultsdataframe:
#             resultsdataframe[filename]= 0
#         for index, row in data_dict.iterrows():        
#             if row["Class"] == "Communication":
#                 sendertypesample, receivertypesample =  cellidlabel[row['sender']],  cellidlabel[row['receiver']] 
#                 if sendertypesample == sendertype and receivertypesample==receivertype:
#                     resultsdataframe[filename]+=row["CCI_score"]
    
    
#     df = pd.DataFrame(list(resultsdataframe.items()), columns=['LRpairs', 'Total_CCI'])
#     df.to_csv(csvfolder+str(sendertype)+"_"+str(receivertype)+'.csv',index=False)
#     print("Done",sendertype,receivertype)

if __name__=="__main__":
    threshold = 0.01
    datasetfolder = "./Dataset/MouseGastrulation/"

    rawdirectorypath = datasetfolder+"/CAGOM_RESULT/"
    
    for dataname in ["E1","E2","E3"]:
        Wholefilepath = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_hightlight_pvalue_csv/"

        singlecelllabelfilepath = datasetfolder+dataname+"_label.csv"
        cellidlabel = utils.getcelllabel(singlecelllabelfilepath,sep = ",")
        totalcelltypes = max(cellidlabel.values())+1
        data_dicts = list_files(Wholefilepath)
        # print(data_dicts)
        csvfolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_celltypeLR_hightlight_pvalue_csv/"
        
        os.makedirs(csvfolder, exist_ok=True)

        for sendertype in tqdm(range(totalcelltypes)):
            for receivertype in  tqdm(range(totalcelltypes)):                
                generateheatmap((csvfolder, data_dicts,sendertype,receivertype))

        # with Pool(10) as p:
        #     tasks = []
        #     # for filename,data_dict in tqdm(data_dicts.items(),desc="Generate heatmap"):   
        #     for sendertype in range(totalcelltypes):
        #         for  receivertype in range(totalcelltypes):
        #             task = (csvfolder, data_dicts,sendertype,receivertype)
        #             tasks.append(task)  
            
        #     pbar = tqdm(total=len(tasks),desc="Building heat map")
        #     def update(*a):
        #         pbar.update()

        #     for task in tasks:                
        #         p.apply_async(generateheatmap, (task,),error_callback=print_error,callback=update)

        #     p.close()
        #     p.join()








            




    
    











