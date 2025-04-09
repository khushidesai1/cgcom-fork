           
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
from scipy.stats import norm

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
    X, Y, kinds = [], [], []
    sl,sr,rl,rr = [] , [] ,[],[]
    sender,receiver = [],[]
    classname = []
    imagefolder,csvfolder, filename, necessary_data, data_dict = args
    ligand, receptor = filename.split("-")
    receptor_data, receptor_avg, ligand_data, ligand_avg, tfs_data, average_value = necessary_data.values()
    tfs = utils.getTFsfromlr(lrTFdf, ligand, receptor)
    tfs_dict = {}
    for tf in tfs:
        if  tf in tfs_data:
            tfs_dict[tf] =[]
            
    for communication_pair in data_dict.keys():
        receptor_cell, ligand_cell = communication_pair.split("-")
        X.append((receptor_data[receptor_cell] + ligand_data[ligand_cell] + ligand_avg + receptor_avg) /
                 (ligand_data[receptor_cell] + receptor_data[ligand_cell] + ligand_avg + receptor_avg))
        Y.append(data_dict[communication_pair])
        sender.append(ligand_cell)
        receiver.append(receptor_cell)
        for tf in tfs:            
            if  tf in tfs_data:                         
                tfs_dict[tf].append(tfs_data[tf][receptor_cell])

        sl.append(ligand_data[ligand_cell])
        sr.append(receptor_data[ligand_cell])
        rl.append(ligand_data[receptor_cell])
        rr.append(receptor_data[receptor_cell])

        kind = "no-com"
        if ligand_data[receptor_cell] < average_value and ligand_data[receptor_cell] < ligand_data[ligand_cell] and receptor_data[receptor_cell]> receptor_data[ligand_cell]:
            for tf in tfs:                    
                if tfs_data[tf][receptor_cell] > 0:
                    kind = "com"                    
                    break
        kinds.append(kind)

    if X:
        mean_X, std_X = np.mean(X), np.std(X)
        mean_Y, std_Y = np.mean(Y), np.std(Y)
        alpha=0.05
        z_threshold = norm.ppf(1 - alpha/2)

        # # Calculate z-scores for X and Y
        X_threshold = mean_X + z_threshold * std_X
        Y_threshold = mean_Y + z_threshold * std_Y

        for x, y, kind in zip(X,Y,kinds):
            
            if x > X_threshold and y> Y_threshold and kind == "com" :
                classname.append("Communication")
            else:
                classname.append("Non-Communication")

                
        data = pd.DataFrame({'sender':sender,'receiver':receiver, "sl": sl, "sr": sr, "rl": rl, "rr": rr,'X': X, 'Y': Y, "kinds": kinds,'Class':classname})
        # Create a DataFrame from the tfs_dict directly (assuming it's a dictionary of lists or arrays)
        tfs_df = pd.DataFrame(tfs_dict)

        # Concatenate the original data DataFrame with the tfs_df DataFrame
        data = pd.concat([data, tfs_df], axis=1)
        data["CCI_score"] = data["X"] * data["Y"]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='X', y='Y', hue='Class', alpha=0.5)
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['X'], data['Y'])
        
        # Creating values for the regression line
        line = slope * data['X'] + intercept

        # Plotting the regression line
        plt.plot(data['X'], line, color='red') # You can change the color if needed

        # avg_x = data['X'].mean()
        # avg_y = data['Y'].mean()

        # Vertical line for average X
        # plt.axvline(x=avg_x, color='blue', linestyle='--', label=f'Average X: {avg_x:.2f}')

        # # Horizontal line for average Y
        # plt.axhline(y=avg_y, color='green', linestyle='--', label=f'Average Y: {avg_y:.2f}')


        # Draw vertical and horizontal lines at the X and Y thresholds
        # plt.axvline(x=X_threshold, color='blue', linestyle='--', label=f'X Threshold: {X_threshold:.2f}')
        # plt.axhline(y=Y_threshold, color='green', linestyle='--', label=f'Y Threshold: {Y_threshold:.2f}')
        data.to_csv(os.path.join(csvfolder,filename +'.csv'))
        # Optionally, add shaded region for intersection
        plt.fill_betweenx([Y_threshold, max(Y)], X_threshold, max(X), color='gray', alpha=0.3, label='Significant Region')

        plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes)

        plt.title('Communication Pair Plot')
        plt.xlabel('X Axis Label')
        plt.ylabel('Y Axis Label')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(imagefolder, filename + "-" + str(round(slope, 4)) + '.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__=="__main__":
    threshold = 0.01
    rawdirectorypath = "./Script/FeaturelevelGAT/tmp/"
    datasetfolder = "./Dataset/MouseGastrulation/"
    # dataname = "E1"
    
    for dataname in ["E1","E2","E3"]:
        Wholefilepath = rawdirectorypath + dataname + "_" + str(threshold) + "_results/"
        
        singlecelldf = utils.loadscfile(datasetfolder + dataname + "_expression.csv")

        lrtffilepath = "./Knowledge/rectome_ligand_receptor_TFs_withReactome.txt"
        lrTFdf = utils.loadlrTF(lrtffilepath,sep="\t")

        imagefolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_hightlight_pvalue_image/"
        
        os.makedirs(imagefolder, exist_ok=True)

        csvfolder = rawdirectorypath + dataname + "_" + str(threshold) + "_results_dot_hightlight_pvalue_csv/"

        os.makedirs(csvfolder, exist_ok=True)

        data_dicts = list_files(Wholefilepath)
        
        with Pool(20) as p:
            tasks = []
            for filename,data_dict in tqdm(data_dicts.items()):                      
                necessary_data = prepare_data(singlecelldf, lrTFdf, filename)
                task = (imagefolder,csvfolder, filename, necessary_data, data_dict)
                tasks.append(task)  
            for task in tqdm(tasks, desc="Generating images"):
                p.apply_async(generateimage, (task,),error_callback=print_error)

            p.close()
            p.join()
   