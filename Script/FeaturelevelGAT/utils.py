
import pandas as pd

import os
import networkx as nx
import math
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from torch_geometric.data import Data, DataLoader


def load_large_pickle(file_path, chunk_size=1024):
    # Get the size of the file in bytes
    file_size = os.path.getsize(file_path)

    # Initialize the progress bar
    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Reading pickle file')

    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Initialize an empty bytes object to store chunks
        file_bytes = b''

        # Read the file in chunks
        for chunk in iter(lambda: file.read(chunk_size), b''):
            file_bytes += chunk
            pbar.update(len(chunk))

    # Close the progress bar
    pbar.close()

    # Load the data from the read bytes
    data = pickle.loads(file_bytes)
    return data


def generate_graph(edges,features, labels):
    
    dataset = []
    numberoflabels= 0
    for node_features, edge_index, label in tqdm(zip(features,edges,labels),desc="Generate graphdata",total=len(labels)):
        numberoflabels = max(numberoflabels,label)
        dataset.append(Data(x=torch.tensor(node_features), edge_index=torch.tensor(edge_index,dtype=torch.int64), y=torch.tensor(label),num_nodes=len(node_features)))
    return dataset, numberoflabels+1
    

def get_file_line_count(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, _ in enumerate(f, 1):
            pass
    return i

def read_csv_with_progress(file_path,chunk_size=1000):
    total_lines =get_file_line_count(file_path) - 1  # Subtract 1 for the header
    pbar = tqdm(total=total_lines, desc="Reading CSV")

    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    df_list = []
    for chunk in chunks:
        df_list.append(chunk)
        pbar.update(chunk.shape[0])

    pbar.close()
    return pd.concat(df_list, ignore_index=False)

def loadscfile(filepath):        
    df = read_csv_with_progress(filepath)
    df.set_index(df.columns[0], inplace=True)
    
    return df

# load TF
def loadtf(file_path):
    tfs = []
    with open(file_path,"r") as tffile:
        for line in tffile.readlines():
            tfs.append(line.strip().upper())
    return tfs

def load_csv_and_create_dict(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    # df = df.str.upper()
    # df['Ligand'] = map(lambda x: str(x).upper(), df['Ligand'])
    # df['Receptor'] = map(lambda x: str(x).upper(), df['Receptor'])
    # df = df.map(lambda x: x.upper() if isinstance(x, str) else x)

    # Check if the dataframe has the required columns
    if 'Ligand' in df.columns and 'Receptor' in df.columns:
        # Create a dictionary where receptor is the key and ligands are values
        receptor_ligand_dict = df.groupby('Receptor')['Ligand'].apply(list).to_dict()
        return receptor_ligand_dict
    else:
        return "Error: CSV file does not have the required 'ligand' and 'receptor' columns."

def generate_sub_dictionary(receptor_ligand_dict, gene_list):
    # Create a sub-dictionary for the given list of genes
    sub_dict = {gene: receptor_ligand_dict[gene] for gene in gene_list if gene in receptor_ligand_dict}
    keys_to_remove = []
    for key, values in sub_dict.items():
        sub_dict[key] = [gene for gene in values if gene in  gene_list]
        if len(sub_dict[key])==0:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del sub_dict[key]
    return sub_dict
    
def pick_random_keys_with_elements(dictionary,genelist, n=25, max_elements=2):
     # Randomly select 'n' keys from the dictionary
    selected_keys = random.sample([gene for gene in list(dictionary.keys()) if gene in genelist], min(n, len(dictionary)))

    # For each selected key, randomly pick up to 'max_elements' from its associated list
    selected_dict = {}
    for key in selected_keys:
        # Filter elements that are in genelist
        filtered_elements = [gene for gene in dictionary[key] if gene in genelist]

        # Determine the number of elements to sample
        num_elements_to_sample = min(max_elements, len(filtered_elements))

        # Sample elements if there are any to sample from
        if num_elements_to_sample > 0:
            selected_dict[key] = random.sample(filtered_elements, num_elements_to_sample)
        # else:
        #     selected_dict[key] = []

    return selected_dict

def pick_random_common_elements(list1, list2, n=10000):
    # Find the common elements between the two lists
    common_elements = set(list1).intersection(list2)

    # If there are enough common elements, randomly pick 'n' of them
    if len(common_elements) >= n:
        return random.sample(common_elements, n)
    else:
        # If not enough common elements, return as many as possible
        return random.sample(common_elements, len(common_elements))

def formsubdataframe(df, listofcolumn):
    return df[listofcolumn]


def eudlidistance(node1,node2):
    return math.dist(node1, node2)


def buildgraph(nodefilelocation,sep="\t",title=False):
    
    G=nx.Graph()
    i = 0    
    locationlist = []
    nodeidlist = []
    minlocation = 9999999999999
    maxlocation = 0    
    # print("start loading node to graph")
    with open(nodefilelocation,"r") as nodefile:
        for line in tqdm(nodefile.readlines(),desc="Loading node to graph"):
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
    for i in tqdm(range(len(locationlist)),desc="Compute disdict"): 
        disdict[i]={}
        for j in range(i+1,len(locationlist)):
            distance = eudlidistance(locationlist[i],locationlist[j])
            disdict[i][j] = distance

    return G,disdict,locationlist,nodeidlist,minlocation,maxlocation

def readdedgestoGraph(G,locationlist,disdict,neighborthresholdratio,minlocation,maxlocation):
    neighborthreshold = minlocation+(maxlocation-minlocation)*neighborthresholdratio
    G.remove_edges_from(list(G.edges()))
    edgelist = []   
    for i in tqdm(range(len(locationlist)),desc="Adding edge"):       
        for j in range(i+1,len(locationlist)):
            distance = disdict[i][j]
            if distance<=neighborthreshold:
                edgelist.append([i,j])
                G.add_edge(i,j)
                
    return G,edgelist

def generate_subgraphs(G):
    subgraphs = {}
    for node in tqdm(G.nodes(),desc="Generate subgraphs"):
        # Get the neighbors of the current node
        neighbors = list(nx.neighbors(G, node))        
        # Create a subgraph with the current node and its neighbors
        if len(neighbors)>0:
            subgraph_nodes = [node] + neighbors
            subgraph = G.subgraph(subgraph_nodes)
            # Store the subgraph
            subgraphs[node] = subgraph

    return subgraphs

def loadlocation(filepath):
    
    return pd.read_csv(filepath, index_col=0)

def loadcelltype(filepath):
    
    return pd.read_csv(filepath, index_col=0,header=None)


def get_adjacency_matrix(G):
    # Get the adjacency matrix in sparse format
    adj_matrix_sparse = nx.adjacency_matrix(G)

    # Convert to a dense format (numpy array)
    adj_matrix_dense = adj_matrix_sparse.toarray()

    return adj_matrix_dense

def generateMaskindex(ligands,lrdictionary):
    # mask = torch.Tensor(p1_channels, p2_channels).fill_(1)  # Correct shape
    # # self.mask[<rows>, <cols>] = 0
    rownumber = 0
    mask_index=[]
    for key, values in lrdictionary.items():
        for value in values:
            # print(key,value)
            columnnumber = ligands.index(value)
            mask_index.append([rownumber,columnnumber])            
        rownumber+=1
    return mask_index

def getcelllabel(filepath,sep = ","):
    cellidlabel={}    
    with open(filepath,'r') as file:
        filelines = file.readlines()
        for line in filelines:
            linedata = line.strip().split(sep)
            cellidlabel[linedata[0]] = int(linedata[1])
    return cellidlabel


def loadlrTF(filepath,sep="\t"):
    ls,rs,tfs = [],[],[]
    
    with open(filepath,'r') as file:
        filelines = file.readlines()
        for line in filelines:
            linedata = line.strip().split(sep)
            for r in linedata[1].split(","):
                ls.append(linedata[0])
                rs.append(r)
                tfs.append(linedata[2])
    return pd.DataFrame({"L":ls,
                         "R":rs,
                         "TF":tfs})

def getTFsfromlr(lrTFdf,ligand,receptor):
    filtered_df = lrTFdf[(lrTFdf['L'] == ligand) & (lrTFdf['R'] == receptor)]   
   
    return filtered_df['TF'].tolist()


def load_data(filepath):
    """Load data from a text file."""
    with open(filepath, 'r') as file:
        lines = file.read().splitlines()[1:]  # Skip the header line
    data_dict = {line.split('\t')[0]: float(line.split('\t')[1]) for line in lines}
    return data_dict

def readcolor():
    numberofcolor = 50
    suffixdatasetname = "./Knowledge/colordict.txt"
    colordict = []
    if os.path.exists(suffixdatasetname):
        with open(suffixdatasetname,"r") as colorfile:
            for line in colorfile.readlines():
                colordict.append(line.strip())
    else:
        with open(suffixdatasetname,"w") as colorfile:
            for i in range(numberofcolor):
                color  = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                colordict.append(color)
                colorfile.write(color+"\n")

    return colordict