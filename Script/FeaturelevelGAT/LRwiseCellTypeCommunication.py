import numpy as np

from scipy import stats
import os
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool


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
    
    # print(args)
    imagefolder,csvfolder, filename, data_dict,cellidlabel = args
    column = max(cellidlabel.values())+1
    significant_matrix = [[0 for _ in range(column)] for _ in range(column)]    
    for index, row in data_dict.iterrows():
        sender, receiver = row['sender'],row['receiver']
        if row["Class"] == "Communication":
            value = significant_matrix[cellidlabel[receiver]][cellidlabel[sender]]
            significant_matrix[cellidlabel[receiver]][cellidlabel[sender]] = value+row["CCI_score"]

    significant_matrix = np.array(significant_matrix)  # Convert to a NumPy array for easier manipulation
    if np.sum(significant_matrix)>0:
        # Calculate the total sum of the matrix
        # total_sum = np.sum(significant_matrix)
        # # Normalize the matrix
        # normalized_matrix = significant_matrix / total_sum
        # print(normalized_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(significant_matrix, annot=False, cmap='viridis')
        # Adding titles and labels (optional)
        plt.title('Significant Communication Heatmap')
        plt.xlabel('Column Label')
        plt.ylabel('Row Label')
        # Show the plot
        plt.savefig(imagefolder+filename+'.png', dpi=300, bbox_inches='tight')
        plt.close()
        np.savetxt(csvfolder+filename+'.csv', significant_matrix, delimiter=",", fmt="%.3f")






if __name__=="__main__":
    threshold = 0.01
    rawdirectorypath = "./Script/FeaturelevelGAT/tmp/"
    datasetfolder = "./Dataset/MouseGastrulation/"

    for dataname in ["E1","E2","E3"]:
        Wholefilepath = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_hightlight_pvalue_csv/"

        singlecelllabelfilepath = datasetfolder+dataname+"_label.csv"
        cellidlabel = utils.getcelllabel(singlecelllabelfilepath,sep = ",")

        data_dicts = list_files(Wholefilepath)
        # print(data_dicts)
        imagefolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_LRcelltype_hightlight_pvalue_image/"
        os.makedirs(imagefolder, exist_ok=True)

        csvfolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_LRcelltype_hightlight_pvalue_csv/"
        os.makedirs(csvfolder, exist_ok=True)


        with Pool(20) as p:
            tasks = []
            for filename,data_dict in tqdm(data_dicts.items(),desc="Generate heatmap"):   
                task = (imagefolder,csvfolder, filename, data_dict,cellidlabel)
                tasks.append(task)  
            
            pbar = tqdm(total=len(tasks),desc="Building heat map")
            def update(*a):
                pbar.update()

            for task in tasks:                
                p.apply_async(generateheatmap, (task,),error_callback=print_error,callback=update)

            p.close()
            p.join()








            




    
    











