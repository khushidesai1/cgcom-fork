           
import numpy as np
import pandas as pd
import os
from glob import glob
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool



def list_files(directory):
    originaldicts = {}
    for filepath in tqdm(glob(os.path.join(directory, '*.txt'))):  # Assuming the files are .txt
        filename = os.path.basename(filepath).split('.')[0]
        with open(filepath, "r") as file:
            lines = file.read().splitlines()[1:]  # Skip the title line
            originaldicts[filename] = {line.split('\t')[0]: float(line.split('\t')[1]) for line in lines}
    return originaldicts

def print_error(value):
    print(value)

def generategraph(filename,data_dict,receptor_dict,imagefolder):
    X, Y = [], []
    receptor, ligand = filename.split("-")

    for communication_pair in data_dict.keys():
        receptorcell, ligandcell = communication_pair.split("-")
        if receptor_dict[receptor][receptorcell] > 0 and receptor_dict[ligand][ligandcell] >0:
            if (receptor_dict[ligand][receptorcell] + receptor_dict[receptor][ligandcell])>0:
                X.append((receptor_dict[receptor][receptorcell] + receptor_dict[ligand][ligandcell]) /
                        (receptor_dict[ligand][receptorcell] + receptor_dict[receptor][ligandcell]))
            
                Y.append(data_dict[communication_pair])
    if len(X)>0:
        data = pd.DataFrame({'X': X, 'Y': Y})
        data['X'].fillna(data['X'].max(), inplace=True)
        plt.figure(figsize=(10, 6))
        
        # Use regplot to add a regression line
        sns.regplot(data=data, x='X', y='Y', scatter_kws={'alpha':0.5})
        
        # Calculate and display the slope
        slope, intercept = np.polyfit(data.dropna()['X'], data.dropna()['Y'], 1)
        plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes)

        plt.title(filename)
        plt.xlabel('The expression ratio')
        plt.ylabel('Attention score')
        plt.grid(True)

        plt.savefig(os.path.join(imagefolder, filename +"-"+str(round(slope,4))+ '.png'), dpi=300, bbox_inches='tight')
        plt.close()



if __name__=="__main__":
    threshold = 0.01
    rawdirectorypath = "./Script/FeaturelevelGAT/tmp/"
    datasetfolder = "./Dataset/MouseGastrulation/"
    dataname = "E3"
    Wholefilepath = rawdirectorypath + dataname + "_" + str(threshold) + "_results/"

    singlecelldf = utils.loadscfile(datasetfolder + dataname + "_expression.csv")
    receptor_dict = singlecelldf.to_dict()

    data_dicts = list_files(Wholefilepath)

    imagefolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_image/"
    os.makedirs(imagefolder, exist_ok=True)

    pbar = tqdm(total=len(data_dicts))
    def update(*a):
        pbar.update()

    p = Pool(10)
    for filename, data_dict in data_dicts.items():
  
        p.apply_async(generategraph,(filename,data_dict,receptor_dict,imagefolder,),error_callback=print_error,callback=update)
    p.close()
    p.join() 
            

