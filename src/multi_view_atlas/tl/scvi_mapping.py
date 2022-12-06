from typing import List, Union

import numpy as np
import scanpy as sc
import scvi
from anndata import AnnData
from scvi.model.base import BaseModelClass

from .map_query import load_query, split_query
from .MultiViewAtlas import MultiViewAtlas

SCVI_TRAIN_PARAMS_DEFAULT = {
    "use_layer_norm": "both",
    "use_batch_norm": "none",
    "encode_covariates": True,
    "dropout_rate": 0.2,
    "n_layers": 2,
}

SCARCHES_TRAIN_PARAMS_DEFAULT = {
    "early_stopping": True,
    "early_stopping_monitor": "elbo_train",
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "plan_kwargs": {"weight_decay": 0.0},
}

SCVI_SETUP_PARAMS_DEFAULT = {}


def _filter_genes_scvi(adata: AnnData, n_top_genes: int = 10000):
    """Filter genes for latent embedding."""
    # Filter genes not expressed anywhere
    sc.pp.filter_genes(adata, min_cells=1)

    # Select HVGs
    if "log1p" not in adata.uns.keys():
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)


def train_scVI(
    adata: AnnData,
    batch_key: str = "donor_id",
    max_epochs: int = 50,
    model_class: BaseModelClass = scvi.model.SCVI,
    n_hvgs: int = 10000,
    train_params: dict = SCVI_TRAIN_PARAMS_DEFAULT,
    model_params: dict = SCVI_SETUP_PARAMS_DEFAULT,
) -> BaseModelClass:
    """Train scvi-tools model and return model object.

    Parameters
    ----------
    adata : AnnData
        AnnData object
    batch_key : str, optional
        batch key, by default "donor_id"
    max_epochs : int, optional
        max number of training epochs, by default 50
    model_class : BaseModelClass, optional
        scvi-tools model class, by default scvi.model.SCVI
    n_hvgs : int, optional
        number of highly variable genes to use, by default 10000
    train_params : dict, optional
        parameters to pass to train method
    model_params : dict, optional
        parameters to pass to model class setup anndata

    Returns
    -------
    BaseModelClass
        scvi-tools model object
    """
    train_adata = adata.copy()
    _filter_genes_scvi(train_adata, n_top_genes=n_hvgs)
    model_class.setup_anndata(train_adata, layer=None, batch_key=batch_key, **model_params)
    model_train = model_class(train_adata, **train_params)
    model_train.train(max_epochs=max_epochs)
    return model_train


def train_scvi_multiview(
    mvatlas: MultiViewAtlas, views: Union[List[str], None] = None, save_path: Union[str, None] = None, **kwargs
):
    """Train scvi-tools model and store view-specific embeddings in MultiViewAtlas object.

    Parameters
    ----------
    mvatlas : MultiViewAtlas
        MultiViewAtlas object
    views : list, optional
        list of views to train scvi model on, by default None (all views)
    save_path : str, optional
        path to save scvi models, by default None (no saving)
    kwargs : dict
        additional arguments passed to multi_view_atlas.tl.scvi_mapping.train_scVI()

    Returns
    -------
    None, modifies MultiViewAtlas in place, adding
    - latent representation for each view in mvatlas.mdata.mod[v].uns['model']
    - scvi model directory in mvatlas.mdata.mod[v].uns['model_dir']
    """
    if views is None:
        views = mvatlas.views
    for view in views:
        model = train_scVI(mvatlas[view], **kwargs)
        mvatlas.mdata.mod[view].obsm[f"X_scVI_{view}"] = model.get_latent_representation()
        # Save model
        if save_path is not None:
            model_dir = f"{save_path}_{view}"
            model.save(model_dir, save_anndata=True, overwrite=True)
            mvatlas.mdata.mod[view].uns["model_dir"] = model_dir


# Query mapping #


def fit_scArches(
    vdata_query: AnnData,
    ref_model_dir: str,
    max_epochs: int = 10,
    model_class: BaseModelClass = scvi.model.SCVI,
    train_kwargs_surgery: dict = SCARCHES_TRAIN_PARAMS_DEFAULT,
) -> BaseModelClass:
    """Fit scArches model to query data.

    Parameters
    ----------
    vdata_query : AnnData
        query data
    ref_model_dir : str
        path to trained reference scvi model
    max_epochs : int, optional
        max number of training epochs, by default 10
    model_class : BaseModelClass, optional
        scvi-tools model class, by default scvi.model.SCVI
    train_kwargs_surgery : dict, optional
        parameters to pass to train method, by default SCARCHES_TRAIN_PARAMS_DEFAULT

    Returns
    -------
    BaseModelClass
        scvi-tools model object
    """
    model_class.prepare_query_anndata(vdata_query, ref_model_dir)
    query_model = model_class.load_query_data(vdata_query, ref_model_dir)
    query_model.train(max_epochs=max_epochs, **train_kwargs_surgery)
    return query_model


def map_scarches_multiview(
    mvatlas_mapped: MultiViewAtlas,
    mvatlas: MultiViewAtlas,
    views: Union[List[str], None] = None,
    save_path: str = None,
    **kwargs,
):
    """Map query to reference with scvi-tools model and store embeddings in MultiViewAtlas object.

    Parameters
    ----------
    mvatlas_mapped : MultiViewAtlas
        MultiViewAtlas object with reference and query data
    mvatlas : MultiViewAtlas
        MultiViewAtlas object with reference data
    views : list, optional
        list of views to train scvi model on, by default None (all views)
    save_path : str, optional
        path to save scvi models, by default None (no saving)
    kwargs : dict
        additional arguments passed to multi_view_atlas.tl.scvi_mapping.fit_scArches()
    """
    if views is None:
        views = mvatlas_mapped.views
    for v in views:
        vdata_query = mvatlas_mapped[v]
        vdata_query = vdata_query[vdata_query.obs["dataset_group"] == "query"].copy()

        # map with scarches
        try:
            ref_model_dir = mvatlas[v].uns["model_dir"]
        except KeyError:
            raise KeyError("missing path to saved model in MultiViewAtlas")

        scarches_model = fit_scArches(vdata_query, ref_model_dir, **kwargs)

        # save dimensions
        X_scVI_query = scarches_model.get_latent_representation()
        X_scVI_atlas = mvatlas[v].obsm[f"X_scVI_{v}"].copy()
        mvatlas_mapped.mdata.mod[v].obsm[f"X_scVI_{v}"] = np.vstack([X_scVI_atlas, X_scVI_query])
        if save_path is not None:
            model_dir = f"{save_path}_{v}"
            scarches_model.save(model_dir, save_anndata=True, overwrite=True)
            mvatlas_mapped.mdata.mod[v].uns["query_model_dir"] = model_dir


def get_views2map(mvatlas_mapped):
    """Find views without view-specific scVI embedding"""
    views2map = []
    for v in mvatlas_mapped.views:
        if "dataset_group" in mvatlas_mapped[v].obs:
            if f"X_scVI_{v}" not in mvatlas_mapped.mdata.mod[v].obsm:
                views2map.append(v)
    return views2map


def map_query_multiview(mvatlas: MultiViewAtlas, adata_query: AnnData, **kwargs) -> MultiViewAtlas:
    """Iteratively map query dataset to MultiViewAtlas dataset

    Params:
    --------
    - mvatlas:
        MultiViewAtlas object of reference (model directories should be stored in mvatlas.mdata.mod[v].uns['model_dir'])
    - adata_query:
        AnnData object of query dataset
    - kwargs:
        additional arguments passed to multi_view_atlas.tl.scvi_mapping.map_scarches_multiview()

    Returns:
    --------
    MultiViewAtlas object of reference + query
    """
    # Check that mvatlas stores path to trained models
    try:
        [mvatlas[v].uns["model_dir"] for v in mvatlas.views]
    except KeyError:
        raise KeyError("Missing path to trained models in mvatlas[view].uns['model_dir']")

    # Mapping to full
    try:
        mvatlas_mapped = load_query(mvatlas, adata_query)
        mvatlas_mapped = split_query(mvatlas_mapped)
    except ValueError:
        query_model = fit_scArches(adata_query, mvatlas["full"].uns["model_dir"], max_epochs=50)
        adata_query.obsm["X_scVI_full"] = query_model.get_latent_representation()
        mvatlas_mapped = load_query(mvatlas, adata_query)
        mvatlas_mapped = split_query(mvatlas_mapped)

    # Iteratively map missing views and propagate
    views2map = get_views2map(mvatlas_mapped)
    while len(views2map) > 0:
        map_scarches_multiview(mvatlas_mapped, mvatlas, views=views2map, **kwargs)
        mvatlas_mapped = split_query(mvatlas_mapped)
        views2map = get_views2map(mvatlas_mapped)
    return mvatlas_mapped


# Embedding #


def knn_umap_multiview(mvatlas: MultiViewAtlas, views: Union[List[str], None] = None, **kwargs):
    """Run KNN graph and UMAP embedding on views of MultiViewAtlas object.

    Params:
    --------
    - mvatlas:
        MultiViewAtlas object
    - views:
        views to run embedding on (default: all views)
    - kwargs:
        additional arguments passed to scanpy.pp.neighbors()
    """
    if views is None:
        views = mvatlas.views
    for view in views:
        sc.pp.neighbors(mvatlas.mdata.mod[view], use_rep=f"X_scVI_{view}", **kwargs)
        sc.tl.umap(mvatlas.mdata.mod[view])
        mvatlas.mdata.mod[view].obsm[f"X_umap_{view}"] = mvatlas.mdata.mod[view].obsm["X_umap"].copy()
        mvatlas.mdata.mod[view].obsm.pop("X_umap")
