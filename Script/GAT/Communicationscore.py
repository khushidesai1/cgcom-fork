import utils
import os
import multiprocessing
import json
from tqdm import tqdm
import pickle
import utils
'''
Given a cellid, a list of edge weight, find the communication with the around cells and reflect the ligand-receptor with number. 

'''
# def print_error(value):
#     print(value)

# def multipooldef(suffixdatasetname,cellid):
#     celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routesocre = utils.loadallmaterialsmallmemory(suffixdatasetname,trainingratio = 0.9)
#     subcellexpression,subsinglecellroute,celllabels,celldistance,celldistanceweight= utils.getconnectedcells(cellid,celllinkweight,singlecelllabels,nozerocellexpression,routesocre,celllocation)
#     totaldict = {}
#     allcells =  list(celllabels.keys())
#     if cellid in allcells:
#         allcells.remove(cellid)
#     for cell2 in allcells:       
#         totaldict[cell2] = utils.findprotentialCombetweentwocells(subcellexpression,subsinglecellroute,celldistance,celldistanceweight,cellid,cell2,lrs)
#     with open(suffixdatasetname+'_GAT/'+cellid+".txt", 'w') as convert_file:
#             convert_file.write(json.dumps(totaldict))

 
             

if __name__=='__main__':
    suffixdatasetnamepre = "./Dataset/MouseGastrulation/" 
    cores = 14
    # suffixdatasetnames = ["./Dataset/MouseGastrulation/E2"]
    for datasetprefixname in ["E1","E2","E3"]:
        suffixdatasetname = suffixdatasetnamepre+datasetprefixname
        outputfile = suffixdatasetname+"_GAT_communication.pkl"
        celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routesocre = utils.loadallmaterialsmallmemory(suffixdatasetname,trainingratio = 0.9)
        
        # if not os.path.exists(outputfile):
        if not os.path.exists(suffixdatasetname+'_GAT/'):
            os.makedirs(suffixdatasetname+'_GAT/')
        
        cores = 14
        pool = multiprocessing.Pool(cores)   
        
        pbar = tqdm(total=len([ cellid for cellid in list(singlecelllabels.keys()) if cellid in celllinkweight]))

        def update(*a):
            pbar.update()   

        for cellid in list(singlecelllabels.keys()):
            if cellid in celllinkweight:
                pool.apply_async(utils.communicationscore, args=(suffixdatasetname,cellid),error_callback=utils.print_error,callback=update)
        pool.close()
        pool.join() 
        i = 0 
        Allcommunication={}
        for filename in tqdm(os.listdir(suffixdatasetname+'_GAT/')):
            f = os.path.join(suffixdatasetname+'_GAT/', filename)
            if os.path.isfile(f):
                cellid = filename.split(".")[0]
                with open(f) as json_file:
                    Allcommunication[cellid] = json.load(json_file)
        with open(outputfile, 'wb') as f:  
            pickle.dump(Allcommunication, f)

        



    



















