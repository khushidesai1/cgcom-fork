import os
import torch
import pickle
import random
import math
import networkx as nx
from utils.utils import (
    get_hyperparameters, convert_anndata_to_df, get_cell_label_dict, build_graph,
    load_csv_and_create_dict, generate_sub_dictionary, loadtf, pick_random_common_elements,
    generateMaskindex, formsubdataframe, getcelllabel, buildgraph, readdedgestoGraph,
    generate_subgraphs, generate_graph
)
from torch_geometric.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from models.model import GATGraphClassifier
from collections import Counter

def train_model(
    hyperparameters,
    dataset_path,
    lr_filepath="./Knowledge/CellPhoneDB_split.csv",
    tf_filepath="./Knowledge/TFlist.txt",
    output_dir="./Script/FeaturelevelGAT/tmp/",
    dataset_name="default",
    labels_key="cell_type"
):
    """
    Train the CGCom model.
    Args:
        dataset_path (str): Path to the dataset.
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training and validation.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        neighbor_threshold_ratio (float): Threshold ratio for building the graph.
        lr_filepath (str): Path to ligand-receptor database.
        tf_filepath (str): Path to transcription factors list.
        output_dir (str): Directory to save outputs.
        dataset_name (str): Name of the dataset.
        labels_key (str): Key for cell type labels in anndata.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    adata = sc.read_h5ad(dataset_path)
    expression_df = convert_anndata_to_df(adata)
    cell_label_dict = get_cell_label_dict(adata, labels_key)
    
    # Load ligand-receptor mapping
    lrmapping = load_csv_and_create_dict(lr_filepath)
    subLRdict = generate_sub_dictionary(lrmapping, list(expression_df.columns))
    
    # Load transcription factors
    TFs = loadtf(tf_filepath)
    selectedTFs = pick_random_common_elements(list(expression_df.columns), TFs)
    
    # Extract ligands and receptors
    ligands = []
    receptors = []
    
    for key, value in subLRdict.items():
        receptors.append(key)
        ligands += value
    ligands = list(set(ligands))
    mask_indexes = generateMaskindex(ligands, subLRdict)
    
    # Form sub-dataframe and normalize
    subdf = formsubdataframe(expression_df, ligands + receptors + selectedTFs)
    scaler = MinMaxScaler()
    subdf = pd.DataFrame(scaler.fit_transform(subdf), index=subdf.index, columns=subdf.columns)
    
    # Build graph from spatial coordinates
    if 'spatial' in adata.obsm:
        cell_locations = adata.obsm['spatial']
        location_df = pd.DataFrame(cell_locations, index=adata.obs_names, columns=['x', 'y'])
        
        # Save location file temporarily
        temp_location_path = f"{output_dir}/temp_location.csv"
        os.makedirs(os.path.dirname(temp_location_path), exist_ok=True)
        location_df.to_csv(temp_location_path)
        
        G, disdict, locationlist, nodeidlist, minlocation, maxlocation = buildgraph(
            temp_location_path, sep=",", title=True
        )
        
        # Clean up temp file
        os.remove(temp_location_path)
    else:
        raise ValueError("Dataset must contain spatial coordinates in adata.obsm['spatial']")
    
    G, edgelist = readdedgestoGraph(G, locationlist, disdict, hyperparameters['neighbor_threshold_ratio'], minlocation, maxlocation)
    
    # Generate subgraphs and features
    orgiganlnodeids = []
    features = []
    edges = []
    labels = []
    
    subgraphs = generate_subgraphs(G)
    for subgraphnode, subg in tqdm(subgraphs.items(), total=len(subgraphs), desc="Creating subgraph"):
        edgelistsource = []
        edgelisttarget = []
        featurelest = []
        
        label = cell_label_dict[nodeidlist[subgraphnode]]
        mainfeature = subdf.loc[nodeidlist[subgraphnode]].tolist()
        featurelest.append(mainfeature)
        
        index = 0
        nodelist = [subgraphnode]
        
        for node in subg.nodes():
            if node != subgraphnode:
                nodelist.append(node)
                mainfeature = subdf.loc[nodeidlist[node]].tolist()
                featurelest.append(mainfeature)
                if index > 0:
                    edgelistsource.append(0)
                    edgelisttarget.append(index)
                index += 1
        
        orgiganlnodeids.append(nodelist)
        features.append(featurelest)
        labels.append(label)
        edges.append([edgelistsource, edgelisttarget])
    
    # Log dataset statistics
    print(f"Number of graphs: {len(features)}") 
    print(f"Number of genes: {len(features[0][0])}")
    print(f"Number of ligands: {len(ligands)}")
    print(f"Number of receptors: {len(subLRdict)}")
    print(f"Number of TFs: {len(selectedTFs)}")
    print(f"Neighbor threshold ratio: {hyperparameters['neighbor_threshold_ratio']}")
    print(f"Train ratio: {hyperparameters['train_ratio']}")
    print(f"Validation ratio: {hyperparameters['val_ratio']}")
    
    # Filter out classes with only one instance
    class_counts = Counter(labels)
    filtered_indices = [i for i, label in enumerate(labels) if class_counts[label] > 1]
    
    # Create filtered dataset
    filtered_features = [features[i] for i in filtered_indices]
    filtered_edges = [edges[i] for i in filtered_indices]
    filtered_labels = [labels[i] for i in filtered_indices]
    filtered_orgiganlnodeids = [orgiganlnodeids[i] for i in filtered_indices]
    
    dataset, num_classes = generate_graph(filtered_edges, filtered_features, filtered_labels)
    
    # Calculate split sizes
    train_size = int(hyperparameters['train_ratio'] * len(dataset))
    valid_size = int(hyperparameters['val_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    # Split dataset
    train_idx, temp_idx, _, temp_labels = train_test_split(
        range(len(dataset)),
        filtered_labels,
        stratify=filtered_labels,
        test_size=valid_size + test_size,
        random_state=42
    )
    
    valid_idx, test_idx = train_test_split(
        temp_idx,
        stratify=[filtered_labels[i] for i in temp_idx],
        test_size=test_size,
        random_state=42
    )
    
    train_dataset = [dataset[i] for i in train_idx]
    valid_dataset = [dataset[i] for i in valid_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    validate_loader = DataLoader(valid_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
    totalloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize the model
    FChidden_channels_2 = 1083
    FChidden_channels_3 = 512
    FChidden_channels_4 = 64
    
    ligand_channel = len(ligands)
    receptor_channel = len(subLRdict)
    TF_channel = len(selectedTFs)
    
    model = GATGraphClassifier(
        FChidden_channels_2=FChidden_channels_2,
        FChidden_channels_3=FChidden_channels_3,
        FChidden_channels_4=FChidden_channels_4,
        num_classes=num_classes,
        device=device,
        ligand_channel=ligand_channel,
        receptor_channel=receptor_channel,
        TF_channel=TF_channel,
        mask_indexes=mask_indexes
    ).to(device)
    
    print(model)
    optimizer = Adam(model.parameters(), lr=hyperparameters['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    # Training function
    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _1, _2, _3 = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    # Testing function
    def test(loader):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out, _1, _2, _3 = model(data.x, data.edge_index, data.batch)
                probs = F.softmax(out, dim=1)
                loss = F.cross_entropy(out, data.y)
                total_loss += loss.item()
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        return total_loss / len(loader.dataset), accuracy, precision, recall, f1, conf_matrix
    
    # Communication recorder function
    def communicationrecorder(outputpath):
        model.eval()
        with torch.no_grad():
            for data, nodelists in tqdm(zip(totalloader, filtered_orgiganlnodeids), total=len(filtered_orgiganlnodeids)):
                data = data.to(device)
                out, communication, attention_coefficients, V = model(data.x, data.edge_index, data.batch)
                firstV = V[0]
                allnodeid = [nodeidlist[i] for i in nodelists]
                
                result = communication * firstV.unsqueeze(0).unsqueeze(-1)
                result = result - result.max()
                firstattention = F.softmax(result, dim=0).cpu()
                
                torch.save(firstattention, outputpath + 'firstattention-' + str(list(nodelists)[0]) + '.pt')
                
                pikcleoutputfile = outputpath + 'nodelist-' + str(list(nodelists)[0]) + '.pkl'
                with open(pikcleoutputfile, 'wb') as f:
                    pickle.dump(list(nodelists), f)
                
                pikcleoutputfile = outputpath + 'allnodeid-' + str(list(nodelists)[0]) + '.pkl'
                with open(pikcleoutputfile, 'wb') as f:
                    pickle.dump(list(allnodeid), f)
    
    # Training loop
    for epoch in range(hyperparameters['num_epochs']):
        train_loss = train()
        train_loss, train_acc, train_precision, train_recall, train_f1, train_conf_matrix = test(train_loader)
        validate_loss, validate_acc, val_precision, val_recall, val_f1, val_conf_matrix = test(validate_loader)
        test_loss, test_acc, test_precision, test_recall, test_f1, test_conf_matrix = test(test_loader)
        
        scheduler.step(validate_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validate Loss: {validate_loss:.4f}, Train Acc: {train_acc:.4f}, Validate Acc: {validate_acc:.4f}, Test Acc: {test_acc:.4f}, LR: {current_lr:.6f}')
    
    # Save model and results
    outputpath = f"{output_dir}/{dataset_name}_{hyperparameters['neighbor_threshold_ratio']}/"
    os.makedirs(outputpath, exist_ok=True)
    
    torch.save(model, outputpath + f'trained_model_{dataset_name}_{hyperparameters['neighbor_threshold_ratio']}.pt')
    
    # Record communication patterns
    communicationrecorder(outputpath)
    
    # Save features information
    pikcleoutputfile = outputpath + f"Feature_{dataset_name}.pkl"
    with open(pikcleoutputfile, 'wb') as f:
        pickle.dump([ligands, receptors, selectedTFs, subLRdict], f)
    
    print(f"Training completed. Model and results saved to {outputpath}")






