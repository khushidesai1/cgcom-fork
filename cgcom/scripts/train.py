import os
import torch
import pickle
from utils import get_hyperparameters, convert_anndata_to_df, get_cell_label_dict, build_graph
import wandb
from torch_geometric.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from model import GATGraphClassifier
from collections import Counter


def train_model(
    dataset_path,
    lr=0.01,
    num_epochs=100,
    batch_size=128,
    train_ratio=0.05,
    val_ratio=0.1,
    neighbor_threshold_ratio=0.01
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
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    hyperparameters = get_hyperparameters(lr, num_epochs, batch_size, train_ratio, val_ratio, neighbor_threshold_ratio)
    expression_df = convert_anndata_to_df(dataset_path)
    cell_label_dict = get_cell_label_dict(expression_df)


LRfilepath = "./Knowledge/CellPhoneDB_split.csv"
lrmapping = utils.load_csv_and_create_dict(LRfilepath)
subLRdict = utils.generate_sub_dictionary(lrmapping, list(singlecelldf.columns))

TFfilepath = "./Knowledge/TFlist.txt"
TFs = utils.loadtf(TFfilepath)
selectedTFs = utils.pick_random_common_elements(list(singlecelldf.columns), TFs)

ligands = []
receptors = []

for key, value in subLRdict.items():
    receptors.append(key)
    ligands+=value
ligands = list(set(ligands))
mask_indexes = utils.generateMaskindex(ligands,subLRdict)


# selectedgenes+=selectedTFs
subdf = utils.formsubdataframe(singlecelldf, ligands+receptors+selectedTFs)
scaler = MinMaxScaler()
subdf = pd.DataFrame(scaler.fit_transform(subdf),index = subdf.index , columns=subdf.columns)

cellidlabel = utils.getcelllabel(singlecelllabelfilepath,sep = ",")

locationfilepath = datasetfolder+dataname+"_location.csv"
G,disdict,locationlist,nodeidlist,minlocation,maxlocation= utils.buildgraph(locationfilepath,sep=",",title=True)

G,edgelist = utils.readdedgestoGraph(G,locationlist,disdict,neighborthresholdratio,minlocation,maxlocation)

orgiganlnodeids=[]
features=[]
edges = []
labels = []
subgraphs = utils.generate_subgraphs(G)
for subgraphnode,subg in tqdm(subgraphs.items(),total=len(subgraphs),desc="Creating subgraph"):
    edgelistsource = []
    edgelisttarget = []    
    featurelest = []
    label = cellidlabel[nodeidlist[subgraphnode]]
    mainfeature = subdf.loc[nodeidlist[subgraphnode]].tolist()
    featurelest.append(mainfeature)
    index = 0
    nodelist = [subgraphnode]
    for node in subg.nodes():
        if not node == subgraphnode:
            nodelist.append(node)
            mainfeature = subdf.loc[nodeidlist[node]].tolist()
            featurelest.append(mainfeature)
            if index>0:
                edgelistsource.append(0)
                edgelisttarget.append(index)
            index+=1
    orgiganlnodeids.append(nodelist)
    features.append(featurelest)
    labels.append(label)
    edges.append([edgelistsource,edgelisttarget])

wandb.run.notes += f"Number of graphs: {len(features)}, Number of genes: {len(features[0][0])}, Number of ligand: {len(ligands)}, Number of receptor: {len(subLRdict)} , Number of TFs: {len(selectedTFs)}, Neighbor threshold ratio:{neighborthresholdratio}, Train ratio:{train_ratio}, Validation ratio:{val_ratio}, Data set: {dataname}"
print(f"Number of graphs: {len(features)}, Number of genes: {len(features[0][0])}, Number of ligand: {len(ligands)}, Number of receptor: {len(subLRdict)} , Number of TFs: {len(selectedTFs)}, Neighbor threshold ratio:{neighborthresholdratio}, Train ratio:{train_ratio}, Validation ratio:{val_ratio} , Data set: {dataname}")


# pikcleoutputfile =  "./Script/FeaturelevelGAT/tmp/test.pkl"
# with open(pikcleoutputfile, 'rb') as f:  
#     G,edges,features, labels,orgiganlnodeids,nodeidlist,cellidlabel,mask_indexes,ligands,subLRdict,selectedTFs  = pickle.load(f)
# receptors = list(subLRdict.keys())
# num_epochs = 1


# mask =  torch.Tensor( len(receptors),len(ligands)).fill_(0) # Correct shape  # Correct shape
# for mask_index in mask_indexes:
#     mask[mask_index[1], mask_index[0]] = 1

# df = pd.DataFrame(mask.numpy(),index =receptors  , columns=ligands)
# df.to_csv("./wmasked.csv")


class_counts = Counter(labels)
# Filter out classes with only one instance
filtered_indices = [i for i, label in enumerate(labels) if class_counts[label] > 1]
# Create filtered dataset
filtered_features = [features[i] for i in filtered_indices]
filtered_edges = [edges[i] for i in filtered_indices]
filtered_labels = [labels[i] for i in filtered_indices]
filtered_orgiganlnodeids = [orgiganlnodeids[i] for i in filtered_indices]
dataset,num_classes = utils.generate_graph(filtered_edges,filtered_features, filtered_labels)

# Calculate split sizes
train_size = int(train_ratio * len(dataset))
valid_size  = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - valid_size


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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
totalloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize the model
FChidden_channels_2 = 1083
FChidden_channels_3 = 512
FChidden_channels_4 = 64

ligand_channel = len(ligands)
receptor_channel = len(subLRdict)
TF_channel = len(selectedTFs)
model = GATGraphClassifier(FChidden_channels_2 =FChidden_channels_2 ,
                            FChidden_channels_3 =FChidden_channels_3 ,
                            FChidden_channels_4 =FChidden_channels_4 ,                            
                            num_classes =num_classes ,device=device,
                            ligand_channel = ligand_channel,
                            receptor_channel = receptor_channel,
                            TF_channel = TF_channel,
                            mask_indexes =mask_indexes).to(device)
# model = torch.nn.DataParallel(model)

print(model)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:  # Replace with your data loader
        data = data.to(device)
        optimizer.zero_grad()
        out,_1,_2,_3= model(data.x, data.edge_index, data.batch)
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
            out,_1,_2,_3 = model(data.x, data.edge_index, data.batch)
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
    return total_loss / len(loader.dataset), accuracy,  precision, recall, f1, conf_matrix

def communicationrecorder(outputpath):
    model.eval()
    subgraphattention=[]    
    with torch.no_grad():
        for data, nodelists in tqdm(zip(totalloader, filtered_orgiganlnodeids),total=len(filtered_orgiganlnodeids)):
            data = data.to(device)            
            out,communication,attention_coefficients,V = model(data.x, data.edge_index, data.batch)
            firstV = V[0]    
            allnodeid = [nodeidlist[i] for i in nodelists]
            # print(allnodeid)
            # exit()
            result = communication*firstV.unsqueeze(0).unsqueeze(-1)
            result = result - result.max()
            firstattention = F.softmax(result,dim=0).cpu()
            matrix_df = pd.DataFrame(firstattention[0,:,:].numpy(), index=receptors, columns=ligands)
            # print(matrix_df)
            # matrix_df.to_csv("./testdf.csv",index=True)
            # exit()
            torch.save(firstattention, outputpath+'firstattention-'+str(list(nodelists)[0])+'.pt')

            pikcleoutputfile = outputpath+'nodelist-'+str(list(nodelists)[0])+'.pkl'
            with open(pikcleoutputfile, 'wb') as f:  
                pickle.dump(list(nodelists), f)
            
            pikcleoutputfile = outputpath+'allnodeid-'+str(list(nodelists)[0])+'.pkl'
            with open(pikcleoutputfile, 'wb') as f:  
                pickle.dump(list(allnodeid), f)

            # subgraphattention.append([firstattention,nodelists])
    # return  subgraphattention     


# Training loop
for epoch in range(num_epochs):
    train_loss = train()
    train_loss, train_acc,  train_precision, train_recall, train_f1, train_conf_matrix = test(train_loader)
    validate_loss, validate_acc,  val_precision, val_recall, val_f1, val_conf_matrix = test(validate_loader)
    test_loss, test_acc,  test_precision, test_recall, test_f1, test_conf_matrix = test(test_loader)
    scheduler.step(validate_loss)
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({
        "Train Loss": train_loss,
        "Validate Loss": validate_loss,
        "Test Loss": test_loss,
        "Train Acc": train_acc,
        "Validate Acc": validate_acc,
        "Test Acc": test_acc,
        "Train Precision": train_precision,
        "Validate Precision": val_precision,
        "Test Precision": test_precision,
        "Train Recall": train_recall,
        "Validate Recall": val_recall,
        "Test Recall": test_recall,
        "Train F1": train_f1,
        "Validate F1": val_f1,
        "Test F1": test_f1,
        "Learning Rate": current_lr
    })        
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validate Loss: {validate_loss:.4f}, Train Acc: {train_acc:.4f}, Validate Acc: {validate_acc:.4f}, Test Acc: {test_acc:.4f}, LR: {current_lr:.6f}')


# wandb.finish()

# dataname="E1"
outputpath = "./Script/FeaturelevelGAT/tmp/"+dataname+"_"+str(neighborthresholdratio)+"/"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
torch.save(model, outputpath+'trained_model_'+dataname+'_'+str(neighborthresholdratio)+'.pt')


subgraphattention = communicationrecorder(outputpath)

pikcleoutputfile = outputpath+"Feature_"+dataname+".pkl"
with open(pikcleoutputfile, 'wb') as f:  
    pickle.dump([ligands,receptors, selectedTFs,subLRdict], f)


# def process_and_save(dffirstattention, receptorid, senderid, outputpath, receptors, ligands):    
#     matrix_df = pd.DataFrame(dffirstattention.numpy(), index=receptors, columns=ligands)
#     matrix_df = matrix_df.round(4)    
#     matrix_df.to_csv(outputpath + "first_attention_" + senderid + "_" + receptorid + ".csv")

# def print_error(value):
#     print(value)

# def update(*a):
#     pbar.update()


# subgraphattention = communicationrecorder()
# outputpath = "./Script/FeaturelevelGAT/tmp/"+dataname+"/"
# if not os.path.exists(outputpath):
#     os.makedirs(outputpath)

# Prepare arguments for parallel processing

# cores = 60


# # Prepare a list to hold the results of apply_async
# results = []

# # Set up tqdm progress bar
# total_tasks = sum(firstattention.size(0) for firstattention, _, _ in subgraphattention)



# for firstattention, attentiocoefficients, nodelists in subgraphattention:
#     allnodeid = [nodeidlist[i] for i in nodelists]
#     senderid = allnodeid[0]
#     pbar = tqdm(total=firstattention.size(0), desc="Processing")
#     pool = multiprocessing.Pool(cores)
#     for i in range(firstattention.size(0)):
#         dffirstattention = firstattention[i]
#         receptorid = allnodeid[i]
#         # Use apply_async and add the result to the list
#         result = pool.apply_async(process_and_save, args=(dffirstattention, receptorid, senderid, outputpath, receptors, ligands), error_callback=print_error, callback=update)
#         results.append(result)

# # Close the pool and wait for the work to finish
#     pool.close()
#     pool.join()

# # Make sure the progress bar is complete
#     pbar.close()

# Use a conservative number of worker processes
# num_workers  # Adjust as needed

# Processing in parallel
# with Pool(num_workers=num_workers) as pool:
#     list(tqdm(pool.imap(process_and_save, args_list), total=len(args_list), desc="Processing"))






