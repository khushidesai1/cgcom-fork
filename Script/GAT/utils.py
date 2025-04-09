
from  model import  GAT
import torch
import pickle
import numpy as np
from tqdm import tqdm
import collections
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
import math
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import numpy as np
import utils
import json
import statistics

def meanvaluep(suffixdatasetname,values,lrkey,i,j,scorelistlr):
    # starttime = time.time()
    output = suffixdatasetname+"_GAT_random_pval/"+str(i)+"_"+str(j)+"_"+lrkey+"_pval.pkl"
    # print(output)
    meanvalue = max(statistics.mean(values),0)
    with open(output, 'wb') as f:  
        pickle.dump(1- (sum(i > meanvalue for i in scorelistlr) / len(scorelistlr)), f)


def loaddataset(suffixdatasetname,datasetprefixname,directed):
    singlecelllabelfilepath = suffixdatasetname+datasetprefixname+"_label.csv"
    cellidlabel = utils.getcelllabel(singlecelllabelfilepath,sep = ",")
    singlecellexpressionfilepath = suffixdatasetname+datasetprefixname+"_expression_median_small.csv"
    locationfilepath = suffixdatasetname+datasetprefixname+"_location.csv"
    singlecellexpression = pd.read_csv(singlecellexpressionfilepath,index_col=  0)
    
    pikcleoutputfile = suffixdatasetname+datasetprefixname+"_raw_sub_graph.pkl"
    if os.path.exists(pikcleoutputfile):
        with open(pikcleoutputfile,'rb') as f:  
            G,disdict,locationlist,nodeidlist,minlocation,maxlocation  = pickle.load(f)
    else:
        G,disdict,locationlist,nodeidlist,minlocation,maxlocation= utils.buildgraph(locationfilepath,sep=",",title=True,directed=directed)
        with open(pikcleoutputfile, 'wb') as f:  
            pickle.dump([G,disdict,locationlist,nodeidlist,minlocation,maxlocation], f)  
    return locationlist,disdict,minlocation,maxlocation,G,singlecellexpression,cellidlabel,nodeidlist

def eudlidistance(node1,node2):
    return math.dist(node1, node2)

def buildgraph(nodefilelocation,sep="\t",title=False,directed=True):
    
    if directed:
        G = nx.DiGraph()
    else:
        G=nx.Graph()
    i = 0    
    locationlist = []
    nodeidlist = []
    minlocation = 9999999999999
    maxlocation = 0    
    # print("start loading node to graph")
    with open(nodefilelocation,"r") as nodefile:
        for line in nodefile.readlines():
            linedata = line.strip().split(sep)
            if title:
                title =False
            else:                
                x = float(linedata[1])
                y = float(linedata[2])
                G.add_node(i,pos=(x,y),label =linedata[0])
                i+=1
                alldistancelist= []
                for location in locationlist:
                    alldistancelist.append(eudlidistance(location,[x,y]))
                maxlocation = max(alldistancelist+[maxlocation])
                minlocation = min(alldistancelist+[minlocation])
                locationlist.append([x,y])
                nodeidlist.append(linedata[0])
    disdict={}
    for i in range(len(locationlist)): 
        disdict[i]={}
        for j in range(i+1,len(locationlist)):
            distance = eudlidistance(locationlist[i],locationlist[j])
            disdict[i][j] = distance
    return G,disdict,locationlist,nodeidlist,minlocation,maxlocation

def readdedgestoGraph(G,locationlist,disdict,neighborthresholdratio,minlocation,maxlocation,directed):
    neighborthreshold = minlocation+(maxlocation-minlocation)*neighborthresholdratio
    G.remove_edges_from(list(G.edges()))
    
    edgelist = []   
    for i in range(len(locationlist)):        
        for j in range(i+1,len(locationlist)):
            distance = disdict[i][j]
            if distance<=neighborthreshold:
                edgelist.append([i,j])
                G.add_edge(i,j)
                if directed:
                    edgelist.append([j,i])
                    G.add_edge(j,i)
    return G,edgelist
    
def getcelllabel(filepath,sep = ","):
    cellidlabel={}    
    with open(filepath,'r') as file:
        filelines = file.readlines()
        for line in filelines:
            linedata = line.strip().split(sep)
            cellidlabel[linedata[0]] = int(linedata[1])
    return cellidlabel

def loadlr(filepath = "./Knowledge/allr.csv",sep=",",title=True,lcol=0,rcol=1):
    lrs = []
    receptordict = {}
    allgeneset = []
    with open(filepath,"r") as LRS:
        lines = LRS.readlines()
        for line in lines:            
            linedata = line.strip().split(sep)
            if title:
                title =False
            else:
                l = linedata[lcol].strip().upper()
                r = linedata[rcol].strip().upper()
                if r in receptordict:
                    receptordict[r].append(l)
                else:
                    receptordict[r] = []
                    receptordict[r].append(l)
                lrs.append((l,r))
                allgeneset.append(l)
                allgeneset.append(r)
    ligandlist = [ lr[0] for lr in lrs]
    allligands = list(set(ligandlist))

    return lrs, allgeneset,receptordict,allligands

def splitetrainingvalidationtestdataset(labels,trainingratio,valratio):
    idx_train=[]
    idx_val =[]
    idx_test = []
    validationlabel = {}
    traininglabel ={}
    traininglabelnumber ={}
    validationlabelnumber ={}

    for label,lencellid in dict(collections.Counter(labels)).items():  
        traininglabelnumber[label] =int(lencellid*trainingratio)
        validationlabelnumber[label] =int(lencellid*valratio) 
    

    nodeids = list(range(len(labels)))
    random.shuffle(nodeids)
    for nodeid in nodeids:
        if labels[nodeid] not in traininglabel:
            traininglabel[labels[nodeid]] = []           
            
        if len(traininglabel[labels[nodeid]])<traininglabelnumber[labels[nodeid]]:
            traininglabel[labels[nodeid]].append(nodeid)  
            idx_train.append(nodeid)

    # print("validation lables")
    random.shuffle(nodeids)
    for nodeid in nodeids:
        if labels[nodeid] not in validationlabel:
            validationlabel[labels[nodeid]] = []   

        if len(validationlabel[labels[nodeid]])<validationlabelnumber[labels[nodeid]]:
            if nodeid not in idx_train:
                validationlabel[labels[nodeid]].append(nodeid)  
                idx_val.append(nodeid)

    for node in range(len(labels)):
        if node not in idx_val and node not in idx_train:
            idx_test.append(node)
    return idx_train,idx_val,idx_test 
    
def generatesubgraph(G,cellidlabel,nodeidlist,singlecellexpression,allligands):
    orgiganlnodeids=[]
    features = []
    edges = []
    labels = []
    for node in tqdm(G.nodes()):
        edgelistsource = []
        edgelisttarget = []
        featurelest = []
        label = cellidlabel[nodeidlist[node]]

        mainfeature = []
        nodeexpressiondict = singlecellexpression[nodeidlist[node]].to_dict()
        for gene in allligands:
            if gene in nodeexpressiondict:
                mainfeature.append(nodeexpressiondict[gene])

        featurelest.append(mainfeature)
        originalnodeid = []
        originalnodeid.append(node)
        if len(G[node]) > 0:
            index = 1
            for nodeid in list(G[node].keys()):
                edgelistsource.append(0)
                edgelisttarget.append(index)
                index+=1
                mainfeature = []
                nodeexpressiondict = singlecellexpression[nodeidlist[nodeid]].to_dict()
                for gene in allligands:
                    if gene in nodeexpressiondict:
                        mainfeature.append(nodeexpressiondict[gene])
                originalnodeid.append(nodeid)
                featurelest.append(mainfeature)

            orgiganlnodeids.append(originalnodeid)    
            edges.append([edgelistsource,edgelisttarget])
            features.append(featurelest)
            labels.append(label)


    return orgiganlnodeids,edges,features,labels


def print_error(value):
    print(value)

def communicationscore(suffixdatasetname,cellid):
    celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routesocre = utils.loadallmaterialsmallmemory(suffixdatasetname,trainingratio = 0.9)
    subcellexpression,subsinglecellroute,celllabels,celldistance,celldistanceweight= utils.getconnectedcells(cellid,celllinkweight,singlecelllabels,nozerocellexpression,routesocre,celllocation)
    totaldict = {}
    allcells =  list(celllabels.keys())
    if cellid in allcells:
        allcells.remove(cellid)
    for cell2 in allcells:       
        totaldict[cell2] = utils.findprotentialCombetweentwocells(subcellexpression,subsinglecellroute,celldistance,celldistanceweight,cellid,cell2,lrs)
    with open(suffixdatasetname+'_GAT/'+cellid+".txt", 'w') as convert_file:
            convert_file.write(json.dumps(totaldict))

def loadlrroute(filepath,sep="\t",title=False,lcol=0,rcol=1,routecol=2):
    lrs = []
    
    allgeneset = []
    with open(filepath,"r") as LRS:
        lines = LRS.readlines()
        for line in lines:            
            linedata = line.strip().split(sep)
            if title:
                title =False
            else:
                l = linedata[lcol].strip().upper()
                r = linedata[rcol].strip().upper()
                route = linedata[routecol].strip().upper()
                allgeneset.append(l)
                allgeneset.append(r)
                lrs.append((l,r,route))
                
    return lrs, list(set(allgeneset))

def replace_negatives(x):
    if x < 0:
        return 0
    else:
        return x

def loadallmaterialsmallmemory(suffixdatasetname,trainingratio = 0.05,randomlabel=False):
    pikcleoutputfile = suffixdatasetname+"_GAT_temp_small.pkl"

    if os.path.exists(pikcleoutputfile):
        with open(pikcleoutputfile,'rb') as f:  
            celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routescoredf  = pickle.load(f)

    else:
        routesocrefilepath = suffixdatasetname+"_routescore.csv"
        singlecellexpressionfilepath = suffixdatasetname+"_expression_median.csv"
        singlecelllabelfilepath = suffixdatasetname+"_label.csv"
        locationfilepath = suffixdatasetname+"_location.csv"

        print("Loading expression")
        singlecellexpression = pd.read_csv(singlecellexpressionfilepath,index_col=  0)

        routescoredf =  pd.read_csv(routesocrefilepath,index_col=  0)

        singlecelllabels,labelsclass = getcelllabel(singlecelllabelfilepath,sep = ",")
        lrfilepath = "./Knowledge/lrtouresall.csv"
        lrs,allgenes = loadlrroute(lrfilepath,sep=",",title=True,lcol=0,rcol=1)
        lrs = [ lr for lr in lrs if lr[0] in list(singlecellexpression.index) and lr[1] in list(singlecellexpression.index) and int(lr[2]) in list(routescoredf.index)]
        singlecellexpression = singlecellexpression[singlecellexpression.index.isin(allgenes)]
        nozerocellexpression = singlecellexpression.applymap(replace_negatives)
        celllocation = pd.read_csv(locationfilepath,index_col=  0)
        linkedgefile = suffixdatasetname+"_GAT_cellcommunication.csv"
        print("Loading loadedgelinkfile")
        celllinkweight = loadedgelinkfile(linkedgefile)
        with open(pikcleoutputfile, 'wb') as f:  
            pickle.dump([celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs,celllocation,routescoredf], f)

    if randomlabel:
        celllinkweight=None
        singlecellexpression=None
        nozerocellexpression=None
        lrs=None
        celllocation=None
        routescoredf=None
        newsinglecelllabels,newlabelsclass={},{}
        selectedableclass = list(labelsclass.keys())
        for key in singlecelllabels.keys():
            randomselected = random.choice(selectedableclass)
            newsinglecelllabels[key] = int(randomselected)
            if randomselected not in newlabelsclass:
                newlabelsclass[randomselected] = []
            newlabelsclass[randomselected].append(key)

        singlecelllabels,labelsclass = newsinglecelllabels,newlabelsclass
        

    return celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routescoredf

def loadedgelinkfile(linkedgefile):
    returnresults={}
    title = True
    with open(linkedgefile,"r") as linkedge:
        for line in linkedge.readlines():
            if title:
                title=False
            else:
                sourceid,targetid,weight = line.strip().split(',')
                if sourceid not in returnresults:
                    returnresults[sourceid] = {}
                returnresults[sourceid][targetid] = float(weight)
    return returnresults

def randomlabels(suffixdatasetname,q):
    scorelist = {}
    celllinkweight,singlecellexpression,nozerocellexpression, singlecelllabels,labelsclass,lrs, celllocation,routesocre = utils.loadallmaterialsmallmemory(suffixdatasetname, 0.9,True)
    outputfile = suffixdatasetname+"_GAT_communication.pkl"
    with open(outputfile,'rb') as f:  # Python 3: open(..., 'rb')
            Allcommunication  = pickle.load(f)
    cellgrouplevelcommunication = {}
    allclass = list(labelsclass.keys())    
    for i in allclass:
        for j in allclass: 
            cellgrouplevelcommunication[i+"_"+j] = utils.celltypelrcommunication(i,j,Allcommunication,labelsclass)
            for lrkey, values in cellgrouplevelcommunication[i+"_"+j].items():
                if lrkey not in scorelist:
                    scorelist[lrkey]=[]
                scorelist[lrkey].append(max(statistics.mean(values),0))
    with open(suffixdatasetname+"_GAT_random_temp/"+str(q)+".pkl", 'wb') as f:  
            pickle.dump(scorelist, f)
