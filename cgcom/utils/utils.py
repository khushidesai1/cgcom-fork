import pandas as pd
import os
import networkx as nx
import math
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import random
from torch_geometric.data import Data

def load_large_pickle(file_path, chunk_size=1024):
    file_size = os.path.getsize(file_path)
    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Reading pickle file')
    with open(file_path, 'rb') as file:
        file_bytes = b''
        for chunk in iter(lambda: file.read(chunk_size), b''):
            file_bytes += chunk
            pbar.update(len(chunk))
    pbar.close()
    data = pickle.loads(file_bytes)
    return data

def generate_graph(edges, features, labels):
    dataset = []
    numberoflabels = 0
    for node_features, edge_index, label in tqdm(zip(features, edges, labels), desc="Generate graphdata", total=len(labels)):
        numberoflabels = max(numberoflabels, label)
        dataset.append(Data(x=torch.tensor(node_features), edge_index=torch.tensor(edge_index, dtype=torch.int64), y=torch.tensor(label), num_nodes=len(node_features)))
    return dataset, numberoflabels + 1

def get_file_line_count(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, _ in enumerate(f, 1):
            pass
    return i

def read_csv_with_progress(file_path, chunk_size=1000):
    total_lines = get_file_line_count(file_path) - 1  # Subtract 1 for the header line (first line)
    pbar = tqdm(total=total_lines, desc="Reading CSV")

    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    df_list = []
    for chunk in chunks:
        df_list.append(chunk)
        pbar.update(chunk.shape[0])

    pbar.close()
    return pd.concat(df_list, ignore_index=False)

def load_transcription_factors(file_path):
    tfs = []
    with open(file_path, "r") as tffile:
        for line in tffile.readlines():
            tfs.append(line.strip().upper())
    return tfs

def load_csv_and_create_dict(file_path):
    df = pd.read_csv(file_path)
    if 'Ligand' in df.columns and 'Receptor' in df.columns:
        receptor_ligand_dict = df.groupby('Receptor')['Ligand'].apply(list).to_dict()
        return receptor_ligand_dict
    else:
        return "Error: CSV file does not have the required 'ligand' and 'receptor' columns."

def generate_sub_dictionary(receptor_ligand_dict, gene_list):
    # Create a sub-dictionary for the given list of genes
    sub_dict = {gene: receptor_ligand_dict[gene] for gene in gene_list if gene in receptor_ligand_dict}
    keys_to_remove = []
    for key, values in sub_dict.items():
        sub_dict[key] = [gene for gene in values if gene in gene_list]
        if len(sub_dict[key]) == 0:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del sub_dict[key]
    return sub_dict


def pick_random_keys_with_elements(dictionary, genelist, n=25, max_elements=2):
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

    return selected_dict


def pick_random_common_elements(list1, list2, n=10000):
    common_elements = set(list1).intersection(list2)
    if len(common_elements) >= n:
        return random.sample(list(common_elements), n)
    else:
        return random.sample(list(common_elements), len(common_elements))


def read_edges_to_graph(G, locationlist, disdict, neighborthresholdratio, minlocation, maxlocation):
    neighborthreshold = minlocation + (maxlocation - minlocation) * neighborthresholdratio
    G.remove_edges_from(list(G.edges()))
    edgelist = []
    for i in tqdm(range(len(locationlist)), desc="Adding edge"):
        for j in range(i + 1, len(locationlist)):
            distance = disdict[i][j]
            if distance <= neighborthreshold:
                edgelist.append([i, j])
                G.add_edge(i, j)

    return G, edgelist

def generate_subgraphs(G):
    subgraphs = {}
    for node in tqdm(G.nodes(), desc="Generate subgraphs"):
        neighbors = list(nx.neighbors(G, node))
        if len(neighbors) > 0:
            subgraph_nodes = [node] + neighbors
            subgraph = G.subgraph(subgraph_nodes)
            subgraphs[node] = subgraph

    return subgraphs

def generate_mask_index(ligands, lrdictionary):
    rownumber = 0
    mask_index = []
    for _, values in lrdictionary.items():
        for value in values:
            columnnumber = ligands.index(value)
            mask_index.append([rownumber, columnnumber])
        rownumber += 1
    return mask_index

def load_lr_tf(filepath, sep="\t"):
    ls, rs, tfs = [], [], []

    with open(filepath, 'r') as file:
        filelines = file.readlines()
        for line in filelines:
            linedata = line.strip().split(sep)
            for r in linedata[1].split(","):
                ls.append(linedata[0])
                rs.append(r)
                tfs.append(linedata[2])
    return pd.DataFrame({"L": ls,
                         "R": rs,
                         "TF": tfs})

def load_data(filepath):
    """Load data from a text file."""
    with open(filepath, 'r') as file:
        lines = file.read().splitlines()[1:]  # Skip the header line
    data_dict = {line.split('\t')[0]: float(line.split('\t')[1]) for line in lines}
    return data_dict


def read_color():
    number_of_color = 50
    suffix_dataset_name = "./Knowledge/colordict.txt"
    colordict = []
    if os.path.exists(suffix_dataset_name):
        with open(suffix_dataset_name, "r") as colorfile:
            for line in colorfile.readlines():
                colordict.append(line.strip())
    else:
        with open(suffix_dataset_name, "w") as colorfile:
            for i in range(number_of_color):
                color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                colordict.append(color)
                colorfile.write(color + "\n")

    return colordict

def get_exp_params(lr=0.01, num_epochs=100, batch_size=128, train_ratio=0.9, val_ratio=0.05, neighbor_threshold_ratio=0.01):
    """
    Initialize the hyperparameters for the CGCom model.
    """
    hyperparameters = {
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "neighbor_threshold_ratio": neighbor_threshold_ratio
    }
    return hyperparameters

def get_model_params(
    fc_hidden_channels_2=1083, 
    fc_hidden_channels_3=512, 
    fc_hidden_channels_4=64, 
    num_classes=10, 
    device="cpu", 
    ligand_channel=1083, 
    receptor_channel=1083, 
    TF_channel=1083, 
    mask_indexes=None, 
    disable_lr_masking=True
):
    model_params = {
        "fc_hidden_channels_2": fc_hidden_channels_2,
        "fc_hidden_channels_3": fc_hidden_channels_3,
        "fc_hidden_channels_4": fc_hidden_channels_4,
        "num_classes": num_classes,
        "device": device,
        "ligand_channel": ligand_channel,
        "receptor_channel": receptor_channel,
        "TF_channel": TF_channel,
        "mask_indexes": mask_indexes,
        "disable_lr_masking": disable_lr_masking
    }
    return model_params

def convert_anndata_to_df(adata):
    """
    Convert anndata to dataframe.
    """
    gene_expression = adata.X.toarray()
    gene_expression_df = pd.DataFrame(gene_expression, index=adata.obs_names, columns=adata.var_names)
    return gene_expression_df


def get_cell_label_dict(adata, labels_key):
    """
    Get the cell label dictionary.
    """
    cell_label_dict = {}
    for cell_id, cell_label in zip(adata.obs_names, adata.obs[labels_key]):
        cell_label_dict[cell_id] = int(cell_label)
    return cell_label_dict


def get_cell_locations_df(adata):
    """
    Get the cell locations.
    """
    cell_locations = adata.obsm["spatial"]
    cell_locations_df = pd.DataFrame(cell_locations, index=adata.obs_names, columns=["X", "Y"])
    return cell_locations_df


def build_graph(cell_locations_df, directed=True):
    """
    Build the graph.
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    i = 0
    locationlist = []
    nodeidlist = []
    minlocation = 9999999999999
    maxlocation = 0
    for location in cell_locations_df[["X", "Y"]].values:
        x = location[0]
        y = location[1]
        G.add_node(i, pos=(x, y))
        i += 1
        alldistancelist = []
        for location in locationlist:
            alldistancelist.append(math.dist(location, [x, y]))
        maxlocation = max(alldistancelist + [maxlocation])
        minlocation = min(alldistancelist + [minlocation])
        locationlist.append([x, y])
        nodeidlist.append(cell_locations_df.index[i-1])
    disdict = {}
    for i in range(len(locationlist)):
        disdict[i] = {}
        for j in range(i + 1, len(locationlist)):
            distance = math.dist(locationlist[i], locationlist[j])
            disdict[i][j] = distance
    return G, disdict, locationlist, nodeidlist, minlocation, maxlocation