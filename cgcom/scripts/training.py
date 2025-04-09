from cgcom.models import GAT
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
import cgcom.utils as utils

def train_gat_model(suffixdatasetname=None, datasetprefixname=None, 
                   neighborthresholdratio=0.005, trainingratio=0.01, 
                   valratio=0.1, lr=0.001, epoches=10, seed=42):
    """
    Train a GAT model on the specified dataset
    
    Parameters:
    -----------
    suffixdatasetname : str
        Path to the dataset directory
    datasetprefixname : str
        Prefix for the dataset files (E1, E2, E3)
    neighborthresholdratio : float
        Ratio for neighbor threshold
    trainingratio : float
        Ratio of data for training
    valratio : float
        Ratio of data for validation
    lr : float
        Learning rate
    epoches : int
        Number of epochs
    seed : int
        Random seed
    """
    if suffixdatasetname is None:
        suffixdatasetname = "./Dataset/MouseGastrulation/"
        
    if datasetprefixname is None:
        datasetprefixnames = ["E1", "E2", "E3"]
    else:
        datasetprefixnames = [datasetprefixname]
        
    lrs, allgeneset, receptordict, allligands = utils.loadlr()
    
    random.seed(seed)
    
    for datasetprefixname in datasetprefixnames:
        directed = True
        locationlist, disdict, minlocation, maxlocation, G, singlecellexpression, cellidlabel, nodeidlist = \
            utils.loaddataset(suffixdatasetname, datasetprefixname, directed) 
            
        G, edgelist = utils.readdedgestoGraph(G, locationlist, disdict, neighborthresholdratio, 
                                            minlocation, maxlocation, directed)
        
        orgiganlnodeids, edges, features, labels = utils.generatesubgraph(G, cellidlabel, 
                                                                       nodeidlist, singlecellexpression, 
                                                                       allligands)
        
        idx_train, idx_val, idx_test = utils.splitetrainingvalidationtestdataset(labels, trainingratio, valratio)

        number_features = len(features[0][0])
        num_hid = number_features
        num_classes = max(labels) + 1

        model = GAT(number_features, num_hid, num_classes).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        def train():
            model.train()
            for idtrain in idx_train:
                feature = torch.FloatTensor(np.array(features[idtrain])).cuda()
                edgeindex = torch.from_numpy(np.array(edges[idtrain])).cuda()
                label = torch.LongTensor(np.array([labels[idtrain]])).cuda()
                out, attentionscore, outputedgelist = model(feature, edgeindex)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        def test():
            model.eval()
            correct = 0
            loss_ = 0
            for idtest in idx_test:
                feature = torch.FloatTensor(np.array(features[idtest])).cuda()
                edgeindex = torch.from_numpy(np.array(edges[idtest])).cuda()
                label = torch.LongTensor(np.array([labels[idtest]])).cuda()
                out, attentionscore, outputedgelist = model(feature, edgeindex)
                loss = criterion(out, label)
                loss_ += loss.item()
                pred = out.argmax(dim=1)
                correct += int((pred == label).sum())
            return correct / len(labels), loss_ / len(labels)

        for epoch in range(epoches):
            train()
            print(f"Epoch {epoch+1}/{epoches}, Test accuracy: {test()[0]:.4f}, Loss: {test()[1]:.4f}")

        # Generate cell communication data
        heatmapdict = {}
        correct = 0
        loss_ = 0
        
        with open(suffixdatasetname+datasetprefixname+"_GAT_cellcommunication.csv", "w") as edgecommunicationfile:
            edgecommunicationfile.write("Cell1,Cell2,linkweight\n")
            
        for edgeindex, feature, originalnodelist, label in tqdm(zip(edges, features, orgiganlnodeids, labels), total=len(labels)):
            model.eval()        
            feature = torch.FloatTensor(np.array(feature)).cuda()
            edgeindex = torch.from_numpy(np.array(edgeindex)).cuda()
            label = torch.LongTensor(np.array([label])).cuda()
            out, attentionweights, outputedgeindex = model(feature, edgeindex)
            pred = out.argmax(dim=1)
            correct += int((pred == label).sum())
            edgeindex = outputedgeindex.tolist()
            edgeweight = attentionweights
            
            for source, target, weight in zip(edgeindex[0], edgeindex[1], edgeweight):
                if not source == target:
                    sendlabel = cellidlabel[nodeidlist[originalnodelist[source]]]
                    targetlabel = cellidlabel[nodeidlist[originalnodelist[target]]]
                    
                    if sendlabel not in heatmapdict:
                        heatmapdict[sendlabel] = {}
                    if targetlabel not in heatmapdict[sendlabel]:
                        heatmapdict[sendlabel][targetlabel] = []
                        
                    with open(suffixdatasetname+datasetprefixname+"_GAT_cellcommunication.csv", "a") as edgecommunicationfile:
                        pline = nodeidlist[originalnodelist[source]]+','+nodeidlist[originalnodelist[target]]+','+str(weight)+'\n'    
                        edgecommunicationfile.write(pline)
                        
                    heatmapdict[sendlabel][targetlabel].append(weight)
                    
        newheatmapdict = {}
        for label1, sendervalue in heatmapdict.items():
            newheatmapdict[label1] = {}
            for label2, receivervalues in sendervalue.items():
                newheatmapdict[label1][label2] = round(sum(receivervalues) / len(receivervalues), 3)

        print(f"Test accuracy: {correct/len(labels)*100:.2f}%")
        newdf = pd.DataFrame(newheatmapdict).sort_index(axis=0).sort_index(axis=1)
        
        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(newdf, annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
        plt.savefig(suffixdatasetname+datasetprefixname+"_GAT_celltypeattention.jpg")
        plt.close()

if __name__ == "__main__":
    train_gat_model() 