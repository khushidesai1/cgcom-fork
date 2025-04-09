import cell2cell as c2c

import numpy as np
import pandas as pd
import scanpy as sc
# import squidpy as sq
# import liana as li

import matplotlib.pyplot as plt
import seaborn as sns


from tqdm.auto import tqdm
import utils

import os
import time

datasetfolder = "./Dataset/MouseGastrulation/"


rawdirectorypath = "./Script/FeaturelevelGAT/tmp/"
log_file_path = rawdirectorypath+"cell2cell_processing_times.log"  # Define log file path
with open(log_file_path, "w") as log_file:  # Open log file in write mode
    log_file.write("Dataset,ProcessingTime(s)\n")  # Write header



for dataname in ["E1","E2","E3"]:
    start_time = time.time()  # Start timing

# for dataname in ["E1"]:
# for dataname in ["E2"]:
# for dataname in ["E3"]:
    print(dataname)
    threshold = 0.01

    outputpath = rawdirectorypath + dataname + "_" + str(threshold) + "_cell2cell/"
        
    os.makedirs(outputpath, exist_ok=True)

    singlecelldf = utils.loadscfile(datasetfolder + dataname + "_expression.csv")
    adata = sc.AnnData(singlecelldf)

    singlecelllabelfilepath = datasetfolder+dataname+"_label.csv"
    labels_data = utils.loadcelltype(singlecelllabelfilepath)
    # print(labels_data)
    adata.obs['label'] = labels_data[2]


    locationfilepath = datasetfolder+dataname+"_location.csv"
    spatial_data  = utils.loadlocation(locationfilepath)
    # print(spatial_data)
    adata.obsm['spatial'] = spatial_data[['X', 'Y']].values 


    adata.layers['counts'] = adata.X.copy()

    adata_og = adata.copy()
    num_bins = 5
    c2c.spatial.create_spatial_grid(adata, num_bins=num_bins)
    lr_pairs = pd.read_csv('./Knowledge/benchmarchlrTF.csv')

    int_columns = ('ligand', 'receptor')
    lr_pairs = c2c.preprocessing.ppi.remove_ppi_bidirectionality(ppi_data=lr_pairs, 
                                                                interaction_columns=int_columns
                                                                )
    ppi_functions = dict()

    for idx, row in lr_pairs.iterrows():
        ppi_label = row[int_columns[0]] + '^' + row[int_columns[1]]
        ppi_functions[ppi_label] = ""

    meta = adata.obs.copy()
    contexts = sorted(meta['grid_cell'].unique())
    context_dict = dict()

    for context in contexts:
        if ('0' in context) or (f'{num_bins-1}' in context):
            context_dict[context] = 'border'
        else:
            context_dict[context] = 'center'

    context_names = contexts
    rnaseq_matrices = []

    for context in tqdm(context_names):
        meta_context = meta.loc[meta['grid_cell'] == context]
        cells = list(meta_context.index)
        
        meta_context.index.name = 'barcode'
        tmp_data = adata[cells]
        # Keep genes in each sample with at least 3 single cells expressing it
        sc.pp.filter_genes(tmp_data, min_cells=3)
        
        # Aggregate gene expression of single cells into cell types
        exp_df = c2c.preprocessing.aggregate_single_cells(rnaseq_data=tmp_data.to_df(),
                                                        metadata=meta_context,
                                                        barcode_col='barcode',
                                                        celltype_col='label',
                                                        method='nn_cell_fraction',
                                                        )
        
        rnaseq_matrices.append(exp_df)

    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=rnaseq_matrices,
                                        ppi_data=lr_pairs,
                                        context_names=context_names,
                                        how='outer',
                                        outer_fraction=1/4.,
                                        complex_sep=None,
                                        interaction_columns=int_columns,
                                        communication_score='expression_mean',
                                        verbose=True
                                        #   device="cuda:0"
                                        )

    meta_tf = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor,
                                                metadata_dicts=[context_dict, ppi_functions, None, None],
                                                fill_with_order_elements=True,
                                                
                                                )

    fig, error = tensor.elbow_rank_selection(upper_rank=20,
                                            runs=20,
                                            init='svd', # If it outputs a memory error, replace by 'random'
                                            automatic_elbow=True,
                                            n_iter_max = 20,
                                            random_state=888,                                            
                                            filename=outputpath+"elbow_rank.jpg",
                                            )
                                             #None # Put a path (e.g. ./Elbow.png) to save the figure
                                            
    # rank=tensor.rank,

    tensor.compute_tensor_factorization(rank=tensor.rank, # tensor.rank # use this instead if you
                                        init='svd', # If it outputs a memory error, replace by 'random'
                                        random_state=888,                                        
                                        verbose=True)

    cmaps = ['plasma', 'Dark2_r', 'tab20', 'tab20']

    fig, axes = c2c.plotting.tensor_factors_plot(interaction_tensor=tensor,
                                             metadata = meta_tf,
                                             sample_col='Element',
                                             group_col='Category',
                                             meta_cmaps=cmaps,
                                             fontsize=14,
                                             filename=outputpath+"tensor_factors_plot.jpg" # Put a path (e.g. ./TF.png) to save the figure
                                            )

    for i in range(tensor.rank):

        tensor.get_top_factor_elements('Ligand-Receptor Pairs', 'Factor {}'.format(i+1), 100).to_csv(outputpath+"Factor_"+str(i)+"_top_factor_element.csv")
        


    tensor.export_factor_loadings(outputpath+'export_factor_loadings.xlsx')

    

    processing_time = time.time() - start_time

    with open(log_file_path, "a") as log_file:  # Open log file in append mode
        log_file.write(f"{dataname},{processing_time}\n")


