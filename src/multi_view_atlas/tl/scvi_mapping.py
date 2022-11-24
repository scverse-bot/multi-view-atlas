import os,sys
import mudata
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData

import scvi
from .map_query import load_query,split_query
from ..utils import check_transition_rule, get_views_from_structure
from .MultiViewAtlas import MultiViewAtlas

## Model training ##

def _filter_genes_scvi(adata, n_top_genes=10000):
    """Filter genes for latent embedding."""
    # Filter genes not expressed anywhere
    sc.pp.filter_genes(adata, min_cells=1)

    # Select HVGs
    if "log1p" not in adata.uns.keys():
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    
def train_scVI(adata, batch_key='donor_id', max_epochs=50, model_class = scvi.model.SCVI):
    arches_params = {
        "use_layer_norm": "both",
        "use_batch_norm": "none",
        "encode_covariates": True,
        "dropout_rate": 0.2,
        "n_layers": 2,
    }

    train_adata = adata.copy()
    _filter_genes_scvi(train_adata)
    model_class.setup_anndata(train_adata, layer=None, batch_key=batch_key)
    model_train = scvi.model.SCVI(train_adata, **arches_params)
    model_train.train(max_epochs=max_epochs)
    return(model_train)

def train_scvi_multiview(mvatlas, views=None, save_path=None, **kwargs):
    '''Train scvi-tools model and store embeddings in MultiViewAtlas object. 
    '''
    if views is None:
        views = mvatlas.views
    for view in views:
        model = train_scVI(mvatlas[view], **kwargs)
        mvatlas.mdata.mod[view].obsm[f'X_scVI_{view}'] = model.get_latent_representation()
        # Save model
        if save_path is not None:
            model_dir = f'{save_path}_{view}'
            model.save(model_dir, save_anndata=True, overwrite=True)
            mvatlas.mdata.mod[view].uns['model_dir'] = model_dir

## Query mapping ## 
            
def fit_scArches(vdata_query, ref_model_dir, max_epochs=100):
    surgery_epochs = max_epochs
    train_kwargs_surgery = {
        "early_stopping": True,
        "early_stopping_monitor": "elbo_train",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "plan_kwargs": {"weight_decay": 0.0},
    }

    scvi.model.SCVI.prepare_query_anndata(vdata_query, ref_model_dir)
    query_model = scvi.model.SCVI.load_query_data(vdata_query, ref_model_dir)
    query_model.train(
        max_epochs=surgery_epochs,
        **train_kwargs_surgery
    )
    return(query_model)

def map_scarches_multiview(mvatlas_mapped, mvatlas, views=None, save_path=None, **kwargs):
    '''Train scvi-tools model and store embeddings in MultiViewAtlas object. 
    '''
    if views is None:
        views = mvatlas_mapped.views
    for v in views:
        vdata_query = mvatlas_mapped[v]
        vdata_query = vdata_query[vdata_query.obs['dataset_group'] == 'query'].copy()

        # map with scarches
        try: 
            ref_model_dir = mvatlas[v].uns['model_dir']
        except KeyError:
            raise KeyError('missing path to saved model in MultiViewAtlas')

        scarches_model = fit_scArches(vdata_query, ref_model_dir, **kwargs)

        # save dimensions
        X_scVI_query = scarches_model.get_latent_representation()
        X_scVI_atlas = mvatlas[v].obsm[f'X_scVI_{v}'].copy()
        mvatlas_mapped.mdata.mod[v].obsm[f'X_scVI_{v}'] = np.vstack([X_scVI_atlas, X_scVI_query])
        if save_path is not None:
            model_dir = f'{save_path}_{v}'
            model.save(model_dir, save_anndata=True, overwrite=True)
            mvatlas_mapped.mdata.mod[v].uns['query_model_dir'] = model_dir
            

def get_views2map(mvatlas_mapped):
    '''Find views without view-specific scVI embedding'''
    view_pairs = mvatlas_mapped.get_view_pairs()
    views2map = []
    for v in mvatlas_mapped.views:
        if 'dataset_group' in mvatlas_mapped[v].obs:
            if f'X_scVI_{v}' not in mvatlas_mapped.mdata.mod[v].obsm:
                views2map.append(v)  
    return(views2map)

def map_query_multiview(
    mvatlas: MultiViewAtlas, 
    adata_query: AnnData
    ) -> MultiViewAtlas:
    '''Iteratively map query dataset to MultiViewAtlas dataset
    
    Params:
    ------
    - mvatlas
    '''
    # Check that mvatlas stores path to trained models
    try:
        [mvatlas[v].uns['model_dir'] for v in mvatlas.views]
    except KeyError:
        raise KeyError("Missing path to trained models in mvatlas[view].uns['model_dir']")
    
    # Mapping to full
    try:
        mvatlas_mapped = load_query(mvatlas, adata_query)
        mvatlas_mapped = split_query(mvatlas_mapped)
    except ValueError:
        query_model = fit_scArches(adata_query, mvatlas['full'].uns['model_dir'], max_epochs=50)
        adata_query.obsm['X_scVI_full'] = query_model.get_latent_representation()
        mvatlas_mapped = load_query(mvatlas, adata_query)
        mvatlas_mapped = split_query(mvatlas_mapped)

    # Iteratively map missing views and propagate
    views2map = get_views2map(mvatlas_mapped)
    while len(views2map) > 0:
        map_scarches_multiview(mvatlas_mapped, mvatlas, views=views2map)
        mvatlas_mapped = split_query(mvatlas_mapped)
        views2map = get_views2map(mvatlas_mapped)
    return(mvatlas_mapped)

## Embedding ## 

def knn_umap_multiview(mvatlas, views=None, **kwargs):
    '''Run KNN graph and UMAP embedding on views of MultiViewAtlas object''' 
    if views is None:
        views = mvatlas.views
    for view in views:
        sc.pp.neighbors(mvatlas.mdata.mod[view], use_rep=f'X_scVI_{view}', **kwargs)
        sc.tl.umap(mvatlas.mdata.mod[view])
        mvatlas.mdata.mod[view].obsm[f'X_umap_{view}'] = mvatlas.mdata.mod[view].obsm['X_umap'].copy()
        mvatlas.mdata.mod[view].obsm.pop('X_umap')   
