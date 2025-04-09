import utils
import pandas as pd
import pickle
from tqdm import tqdm
import multiprocessing
import os
import statistics




if __name__=='__main__':   
    suffixdatasetnamepre = "./Dataset/MouseGastrulation/" 
    
    for datasetprefixname in ["E1","E2","E3"]:
        cores = 2
        iterationnumber = 100
        suffixdatasetname = suffixdatasetnamepre+datasetprefixname
        if not os.path.exists(suffixdatasetname+"_GAT_random_temp/"):
            os.makedirs(suffixdatasetname+"_GAT_random_temp/")
        pikcleoutputfile = suffixdatasetname+"_GAT_randomlabelmean.pkl"
        pool = multiprocessing.Pool(cores)   
        for q in range(iterationnumber):            
            pbar = tqdm(total=iterationnumber)
            def update(*a):
                pbar.update()
            pool.apply_async(utils.randomlabels, args=(suffixdatasetname,q),error_callback=utils.print_error,callback=update) 
        pool.close()
        pool.join()     
        totalscoredict = {}
        for filename in tqdm (os.listdir(suffixdatasetname+"_GAT_random_temp/")):
            f = os.path.join(suffixdatasetname+"_GAT_random_temp/", filename)
            if os.path.isfile(f):
                with open(f,'rb') as fpkl:  # Python 3: open(..., 'rb')
                    scoredict  = pickle.load(fpkl)
                    for key, vallist in scoredict.items():
                        if key not in totalscoredict:
                            totalscoredict[key] = []
                        totalscoredict[key] +=vallist
        with open(pikcleoutputfile, 'wb') as f:  
            pickle.dump(totalscoredict, f)
                        

        if not os.path.exists(suffixdatasetname+"_GAT_random_pval/"):
            os.makedirs(suffixdatasetname+"_GAT_random_pval/")

        celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routesocre = utils.loadallmaterialsmallmemory(suffixdatasetname,trainingratio = 0.9)
        
        outputfile = suffixdatasetname+"_GAT_communication.pkl"
        with open(outputfile,'rb') as f:  # Python 3: open(..., 'rb')
                Allcommunication  = pickle.load(f)
        cellgrouplevelcommunication = {}
        allclass = list(labelsclass.keys())
        # load random
        pikcleoutputfile = suffixdatasetname+"_GAT_randomlabelmean.pkl"
        with open(pikcleoutputfile,'rb') as f:  # Python 3: open(..., 'rb')
            scorelist  = pickle.load(f)
        cores = 10
        for i in tqdm(allclass):
            for j in tqdm(allclass):                    
                cellgrouplevelcommunication[i+"_"+j] = utils.celltypelrcommunication(i,j,Allcommunication,labelsclass)
                pool = multiprocessing.Pool(cores) 
                for lrkey, values in cellgrouplevelcommunication[i+"_"+j].items():
                    pool.apply_async(utils.meanvaluep, args=(suffixdatasetname,values,lrkey,i,j,scorelist[lrkey]),error_callback=utils.print_error) 
                pool.close()
                pool.join()
                cellgrouplevelcommunication[i+"_"+j][lrkey] = 1- (sum(i > max(statistics.mean(values),0) for i in scorelist[lrkey]) / len(scorelist[lrkey]))
        cellgrouplevelcommunication={}
        for filename in tqdm (os.listdir(suffixdatasetname+"_GAT_random_pval/")):
            if len(filename.split("_"))>5: 
                i,j,l,r,route = filename.split("_")[0],filename.split("_")[1],filename.split("_")[2],filename.split("_")[3],filename.split("_")[4]
                lrkey = l+"_"+r+"_"+route
                if i+"_"+j not in cellgrouplevelcommunication:
                    cellgrouplevelcommunication[i+"_"+j] = {}
                f = os.path.join(suffixdatasetname+"_GAT_random_pval/", filename)
                if os.path.isfile(f):
                    with open(f,'rb') as fpkl:  
                        cellgrouplevelcommunication[i+"_"+j][lrkey]  = pickle.load(fpkl)
        with open(suffixdatasetname+"_GAT_permutation.pkl", 'wb') as f:  
                pickle.dump(cellgrouplevelcommunication, f)                
        if not os.path.exists(suffixdatasetname+"_communicationsummary/"):
            os.makedirs(suffixdatasetname+"_communicationsummary/")
        singlecelllabelfilepath = suffixdatasetname+"_label.csv"
        cellgroupname = utils.loadcellgroupname(singlecelllabelfilepath)
        celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routesocre = utils.loadallmaterialsmallmemory(suffixdatasetname,0.9)
        outputfile = suffixdatasetname+"_GAT_communication.pkl"
        with open(outputfile,'rb') as f:  
                Allcommunication  = pickle.load(f)
        cellgrouplevelcommunication = {}
        allclass = list(labelsclass.keys())
        realcellgrouplevelcommunication={
            "ligand":[],
            "receptor":[],
            "down stream route":[],
            "source group":[],
            "target group":[],            
            "communcation score":[],
            "pvalues":[],
            "Celltype Combine":[],
            "LR Combine":[]
        }
        with open(suffixdatasetname+"_GAT_permutation.pkl", 'rb') as f:  
            pvaluedf  = pickle.load(f)
        for i in tqdm(allclass):
            for j in allclass:
                for lrkey, values in utils.celltypelrcommunication(i,j,Allcommunication,labelsclass).items():
                    if lrkey in pvaluedf[i+"_"+j]:
                        if 1 - pvaluedf[i+"_"+j][lrkey]<0.05 and statistics.mean(values)>0 and not i==j:
                            ligand,receptor,route = lrkey.split("_")
                            realcellgrouplevelcommunication["ligand"].append(ligand)
                            realcellgrouplevelcommunication["receptor"].append(receptor)
                            realcellgrouplevelcommunication["down stream route"].append(route)
                            realcellgrouplevelcommunication["source group"].append(cellgroupname[i])
                            realcellgrouplevelcommunication["target group"].append(cellgroupname[j])
                            realcellgrouplevelcommunication["communcation score"].append(max(statistics.mean(values),0))
                            realcellgrouplevelcommunication["pvalues"].append(1-pvaluedf[i+"_"+j][lrkey])
                            realcellgrouplevelcommunication["Celltype Combine"].append(cellgroupname[i]+"_"+cellgroupname[j])
                            realcellgrouplevelcommunication["LR Combine"].append(ligand+"_"+receptor)
        pd.DataFrame(realcellgrouplevelcommunication).to_csv(suffixdatasetname+"_GAT_CGCom.csv")

