import numpy as np
import pytest
import scvi.model

from multi_view_atlas.tl import MultiViewAtlas, scvi_mapping
from multi_view_atlas.utils import get_views_from_structure, sample_dataset


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("models")
    return save_dir


@pytest.fixture
def query_reference():
    adata = sample_dataset()
    # Reduce views for faster training
    adata.uns["view_hierarchy"]["lymphoid"] = {"NKT cells": None}
    adata.obsm["view_assign"] = adata.obsm["view_assign"][get_views_from_structure(adata.uns["view_hierarchy"])].copy()
    batches = np.random.choice(["B1", "B2", "B3"], size=adata.n_obs)
    adata.obs["batch"] = batches
    query_cells = adata.obs_names[adata.obs["batch"] == "B3"]
    adata_query = adata[query_cells].copy()
    adata = adata[~adata.obs_names.isin(query_cells)].copy()
    mva = MultiViewAtlas(adata, transition_rule="X_pca")
    return (mva, adata_query)


# Test that mapping works with different types of models


def test_scvi_output(query_reference, model_dir):
    mva, _ = query_reference
    scvi_mapping.train_scvi_multiview(mva, batch_key="batch", save_path=model_dir, max_epochs=1)
    for v in mva.mdata.mod.keys():
        assert f"X_scVI_{v}" in mva.mdata[v].obsm
        assert "model_dir" in mva.mdata[v].uns


# @pytest.mark.skip(reason="Takes a long time")


@pytest.mark.skip(reason="Takes a long time and scANVI is broken")
def test_scanvi_mapping(query_reference, model_dir):
    mva, adata_query = query_reference
    scvi_mapping.train_scvi_multiview(
        mva,
        batch_key="batch",
        save_path=model_dir,
        max_epochs=1,
        model_class=scvi.model.SCANVI,
        model_params={"labels_key": "louvain", "unlabeled_category": "Unknown", "extend_categories": True},
    )
    mva_mapped = scvi_mapping.map_query_multiview(mva, adata_query, model_class=scvi.model.SCANVI)
    for v in mva_mapped.mdata.mod.keys():
        assert f"X_scVI_{v}" in mva_mapped.mdata[v].obsm
        assert "model_dir" in mva_mapped.mdata[v].uns
