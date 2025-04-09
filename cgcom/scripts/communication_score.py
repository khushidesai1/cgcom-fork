import os
import multiprocessing
import json
from tqdm import tqdm
import pickle
import cgcom.utils as utils

def analyze_cell_communication(suffixdatasetname=None, cores=14):
    """
    Analyze cell communication using the GAT model results
    
    Parameters:
    -----------
    suffixdatasetname : str
        Path to the dataset directory
    cores : int
        Number of CPU cores to use
    """
    if suffixdatasetname is None:
        suffixdatasetnamepre = "./Dataset/MouseGastrulation/"
        datasetprefixnames = ["E1", "E2", "E3"]
    else:
        suffixdatasetnamepre = suffixdatasetname
        datasetprefixnames = [""]

    for datasetprefixname in datasetprefixnames:
        if datasetprefixname:
            suffixdatasetname = suffixdatasetnamepre + datasetprefixname
        else:
            suffixdatasetname = suffixdatasetnamepre
            
        outputfile = suffixdatasetname + "_GAT_communication.pkl"
        
        celllinkweight, singlecellexpression, nozerocellexpression, singlecelllabels, labelsclass, lrs, celllocation, routesocre = \
            utils.loadallmaterialsmallmemory(suffixdatasetname, trainingratio=0.9)
        
        if not os.path.exists(suffixdatasetname + '_GAT/'):
            os.makedirs(suffixdatasetname + '_GAT/')
        
        pool = multiprocessing.Pool(cores)
        
        pbar = tqdm(total=len([cellid for cellid in list(singlecelllabels.keys()) if cellid in celllinkweight]))

        def update(*a):
            pbar.update()

        for cellid in list(singlecelllabels.keys()):
            if cellid in celllinkweight:
                pool.apply_async(utils.communicationscore, 
                                args=(suffixdatasetname, cellid), 
                                error_callback=utils.print_error,
                                callback=update)
        
        pool.close()
        pool.join()
        
        Allcommunication = {}
        for filename in tqdm(os.listdir(suffixdatasetname + '_GAT/')):
            f = os.path.join(suffixdatasetname + '_GAT/', filename)
            if os.path.isfile(f):
                cellid = filename.split(".")[0]
                with open(f) as json_file:
                    Allcommunication[cellid] = json.load(json_file)
        
        with open(outputfile, 'wb') as f:
            pickle.dump(Allcommunication, f)
        
        print(f"Cell communication analysis complete for {suffixdatasetname}")

if __name__ == '__main__':
    analyze_cell_communication() 