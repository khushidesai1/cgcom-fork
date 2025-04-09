           
import numpy as np
import pandas as pd
import os
from glob import glob
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool


def list_files(directory):
    originaldicts = {}
    for filepath in tqdm(glob(os.path.join(directory, '*.txt'))):  # Assuming the files are .txt
        filename = os.path.basename(filepath).split('.')[0]
        ligand, receptor = filename.split("-")
        tfs = utils.getTFsfromlr(lrTFdf, ligand, receptor)
        if tfs: 
            with open(filepath, "r") as file:
                lines = file.read().splitlines()[1:]  # Skip the title line
                originaldicts[filename] = {line.split('\t')[0]: float(line.split('\t')[1]) for line in lines}

    return originaldicts


def print_error(value):
    print("error")
    print(value)


def prepare_data(singlecelldf, lrTFdf, filename):
    """
    Prepare the necessary data for a single task.
    This function extracts relevant columns from singlecelldf based on the filename.
    """
    ligand, receptor = filename.split("-")
    tfs = utils.getTFsfromlr(lrTFdf, ligand, receptor)
    necessary_data = {
        'receptor': singlecelldf[receptor].to_dict(),
        'receptor_avg': singlecelldf[receptor].mean(),
        'ligand': singlecelldf[ligand].to_dict(),
        'ligand_avg': singlecelldf[ligand].mean(),
        'tfs': {tf: singlecelldf[tf].to_dict() for tf in tfs if tf in singlecelldf},
        'averagevalue': singlecelldf[ligand].mean()
    }
    return necessary_data



def generateimage(args):

    X,Y,kinds = [],[],[] 
    
    imagefolder,csvfolder, filename, necessary_data, data_dict = args
    ligand, receptor = filename.split("-")
    # Extract preprocessed necessary data
    receptor_data,receptoravg, ligand_data,ligandavg, tfs_data, averagevalue = necessary_data.values()
           
    tfs  = utils.getTFsfromlr(lrTFdf,ligand,receptor)
    tfs_dict = {}
    for tf in tfs:
        if  tf in tfs_data:
            tfs_dict[tf] =[]
       
    for communication_pair in data_dict.keys():
        receptorcell, ligandcell = communication_pair.split("-")
                            
        X.append((receptor_data[receptorcell] + ligand_data[ligandcell] +ligandavg+receptoravg) /
                (ligand_data[receptorcell] + receptor_data[ligandcell] +ligandavg+receptoravg ))
        Y.append(data_dict[communication_pair])
        # print(1)
        for tf in tfs:            
            if  tf in tfs_data: 
                        
                tfs_dict[tf].append(tfs_data[tf][receptorcell])
        
        kind = "no-com"
        if ligand_data[receptorcell] < averagevalue and ligand_data[receptorcell] <ligand_data[ligandcell]:
            for tf in tfs: 
                if  tf in tfs_data:                                
                    if tfs_data[tf][receptorcell]>0:
                        kind="com"  
                        break
        
        kinds.append(kind)
    if X:

        data = pd.DataFrame({'X': X, 'Y': Y, "kinds": kinds})
        
        # Create a DataFrame from the tfs_dict directly (assuming it's a dictionary of lists or arrays)
        tfs_df = pd.DataFrame(tfs_dict)

        # Concatenate the original data DataFrame with the tfs_df DataFrame
        data = pd.concat([data, tfs_df], axis=1)
        data.to_csv(os.path.join(csvfolder, filename+ '.csv'))
        plt.figure(figsize=(10, 6))

        # Plotting the scatter points with colors based on 'kinds'
        sns.scatterplot(data=data, x='X', y='Y', hue='kinds', alpha=0.5)

        # Calculating the regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['X'], data['Y'])

        # Creating values for the regression line
        line = slope * data['X'] + intercept

        # Plotting the regression line
        plt.plot(data['X'], line, color='red') # You can change the color if needed

        avg_x = data['X'].mean()
        avg_y = data['Y'].mean()

        # Vertical line for average X
        plt.axvline(x=avg_x, color='blue', linestyle='--', label=f'Average X: {avg_x:.2f}')

        # Horizontal line for average Y
        plt.axhline(y=avg_y, color='green', linestyle='--', label=f'Average Y: {avg_y:.2f}')

        plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes)

        plt.title('Communication Pair Plot')
        plt.xlabel('X Axis Label')
        plt.ylabel('Y Axis Label')
        plt.grid(True)

        plt.legend()
        plt.savefig(os.path.join(imagefolder, filename +"-"+str(round(slope,4))+ '.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__=="__main__":
    threshold = 0.01
    rawdirectorypath = "./Script/FeaturelevelGAT/tmp/"
    datasetfolder = "./Dataset/MouseGastrulation/"
    # dataname = "E1"
    
    for dataname in ["E1","E2","E3"]:
        print(dataname)
        Wholefilepath = rawdirectorypath + dataname + "_" + str(threshold) + "_results/"
        
        singlecelldf = utils.loadscfile(datasetfolder + dataname + "_expression.csv")

        lrtffilepath = "./Knowledge/rectome_ligand_receptor_TFs_withReactome.txt"
        lrTFdf = utils.loadlrTF(lrtffilepath,sep="\t")

        imagefolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_hightlight_image/"
        os.makedirs(imagefolder, exist_ok=True)

        
        data_dicts = list_files(Wholefilepath)
        
        with Pool(20) as p:
            tasks = []
            for filename,data_dict in tqdm(data_dicts.items()):                      
                necessary_data = prepare_data(singlecelldf, lrTFdf, filename)
                task = (imagefolder, filename, necessary_data, data_dict)
                tasks.append(task)  
            for task in tqdm(tasks, desc="Generating images"):
                p.apply_async(generateimage, (task,), error_callback=print_error)

            p.close()
            p.join()
   