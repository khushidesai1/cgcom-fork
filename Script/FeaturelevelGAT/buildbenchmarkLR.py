
import numpy as np

from scipy import stats
import os
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool


def list_files(directory):
    
    originaldictionarys = {}
    for filename in tqdm(os.listdir(directory),desc="readfile"):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            df = pd.read_csv(path,index_col = 0)
            originaldictionarys[filename.split(".")[0]] = df
        
    return originaldictionarys


if __name__=="__main__":
    dataname= "E1"
    threshold = 0.01
    
    rawdirectorypath = "./Script/FeaturelevelGAT/tmp/"
    Wholefilepath = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_hightlight_pvalue_csv/"

    with open("./Knowledge/benchmarchlrTF.csv", "w") as output:
        output.write("ligand,receptor\n")
        for filename in tqdm(os.listdir(Wholefilepath),desc="readfile"):
            path = os.path.join(Wholefilepath, filename)
            if os.path.isfile(path):
                
                ligand,receptor = filename.split(".")[0].split("-")
                output.write(ligand+","+receptor+"\n")
                