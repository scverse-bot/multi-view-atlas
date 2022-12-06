import mudata
import pytest
import scanpy as sc

from multi_view_atlas.pl import multiview_embedding
from multi_view_atlas.tl import MultiViewAtlas
from multi_view_atlas.utils import sample_dataset


def test_multiview_embedding_basis():
    adata = sample_dataset()
    adata_dict = {}
    adata_dict["full"] = adata.copy()
    view_assign = adata.obsm["view_assign"].copy()
    for v in view_assign:
        adata_dict[v] = adata[view_assign[v] == 1].copy()
    for v in adata_dict.keys():
        sc.pp.normalize_total(adata_dict[v], target_sum=1e4)
        sc.pp.log1p(adata_dict[v])
        sc.pp.highly_variable_genes(adata_dict[v], n_top_genes=1000)
        sc.pp.pca(adata_dict[v], n_comps=10)

    mdata = mudata.MuData(adata_dict)
    mvatlas = MultiViewAtlas(mdata, transition_rule="louvain", rename_obsm=True)
    try:
        multiview_embedding(mvatlas, view="NKT cells", basis_from_full=False, basis="X_pca")
    except KeyError:
        raise AssertionError("multiview_embedding throws an exception")

    mdata = mudata.MuData(adata_dict)
    mvatlas = MultiViewAtlas(mdata, transition_rule="louvain", rename_obsm=False)
    try:
        multiview_embedding(mvatlas, view="NKT cells", basis_from_full=True, basis="X_pca")
    except KeyError:
        raise AssertionError("multiview_embedding throws an exception")
    with pytest.raises(KeyError):
        multiview_embedding(mvatlas, view="NKT cells", basis_from_full=False, basis="X_pca")
