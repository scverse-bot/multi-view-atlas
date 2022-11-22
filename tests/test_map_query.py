import numpy as np

from multi_view_atlas.tl import MultiViewAtlas
from multi_view_atlas.tl.map_query import add_query
from multi_view_atlas.utils import sample_dataset


def test_mapping_output():
    """Test that consecutive mapping is the same as mapping in one step"""
    adata = sample_dataset()
    # split in query and atlas
    query_cells = np.random.choice(adata.obs_names, size=int(np.round(adata.n_obs * 0.1)), replace=False)
    adata_query = adata[query_cells].copy()
    adata = adata[~adata.obs_names.isin(query_cells)].copy()
    mva = MultiViewAtlas(adata, transition_rule="louvain")
    mva_mapped = add_query(mva, adata_query)
    assert mva_mapped.mdata.n_obs > mva.mdata.n_obs, "missing query cells"
    assert [
        (mva_mapped.mdata[v].obs["dataset_group"] == "query").sum() > 0 for v in mva_mapped.views
    ], "missing query cells"
    assert (
        mva_mapped.mdata["T cells"].obs["louvain"].isin(["CD4 T cells", "CD8 T cells"]).all()
    ), "wrong assignment of query cells"


def test_consecutive_mapping():
    """Test that consecutive mapping is the same as mapping in one step"""
    adata = sample_dataset()
    # split in query and atlas
    query_cells = np.random.choice(adata.obs_names, size=int(np.round(adata.n_obs * 0.1)), replace=False)
    adata_query = adata[query_cells].copy()
    adata = adata[~adata.obs_names.isin(query_cells)].copy()
    mva = MultiViewAtlas(adata)

    # New embedding for NKT view
    v = "NKT cells"
    vadata = mva[v]
    vadata.obsm[f"X_pca_{v}"] = vadata.obsm["X_pca"].copy()
    mva.mdata[v].obsm = vadata.obsm.copy()

    # Change transition rule to use NKT specific PCA
    mva.set_transition_rule("NKT cells", "T cells", "X_pca_NKT cells")

    # Map in one go (all embeddings present)
    adata_query_1 = adata_query.copy()
    adata_query_1.obsm["X_pca_NKT cells"] = adata_query.obsm["X_pca"].copy()
    mva_1 = add_query(mva, adata_query_1)

    # Map in 2 consecutive steps (adding embedding in the middle)
    adata_query_2 = adata_query.copy()
    mva_2 = add_query(mva, adata_query_2)
    assert mva_1.mdata.obsm["view_assign"]["T cells"].sum() > mva_2.mdata.obsm["view_assign"]["T cells"].sum()
    v = "NKT cells"
    vdata = mva_2[v]
    mva_2.mdata.mod[v].obsm[f"X_pca_{v}"] = vdata.obsm["X_pca"].copy()
    adata_query_2.obsm["X_pca_NKT cells"] = adata_query_2.obsm["X_pca"].copy()
    mva_2 = add_query(mva_2, adata_query_2)
    assert all(
        mva_1.mdata[v].obs_names == mva_2.mdata[v].obs_names
    ), "Consecutive mapping should be the same as mapping in one step"
