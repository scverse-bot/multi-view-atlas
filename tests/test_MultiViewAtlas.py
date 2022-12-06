import mudata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from multi_view_atlas.tl import MultiViewAtlas
from multi_view_atlas.utils import (
    get_parent_view,
    get_views_from_structure,
    sample_dataset,
)


def test_X():
    adata = sample_dataset()
    mva = MultiViewAtlas(adata)
    assert mva.mdata["full"].X is not None, "X is None"
    assert mva.mdata["full"].X.shape == adata.X.shape, "X shape is not correct"


def test_init_from_adata():
    adata = sample_dataset()
    mva = MultiViewAtlas(adata, subset_obsm=False)
    assert "X_pca" not in mva.mdata["T cells"].obsm
    mva = MultiViewAtlas(adata, subset_obsm=True)
    assert "X_pca" in mva.mdata["T cells"].obsm


def test_init_from_mudata():
    adata = sample_dataset()
    view_assign = adata.obsm["view_assign"].copy()
    adata_dict = {}
    adata_dict["full"] = adata.copy()
    for v in view_assign:
        adata_dict[v] = adata[view_assign[v] == 1].copy()
    mdata = mudata.MuData(adata_dict)
    mvatlas = MultiViewAtlas(mdata, transition_rule="louvain")
    assert [k.endswith("lymphoid") for k in mvatlas.mdata["lymphoid"].obsm.keys()]


def test_keep_slots():
    adata = sample_dataset()
    adata.layers["counts"] = adata.X.copy()
    view_assign = adata.obsm["view_assign"].copy()
    adata_dict = {}
    adata_dict["full"] = adata.copy()
    for v in view_assign:
        adata_dict[v] = adata[view_assign[v] == 1].copy()
    mdata = mudata.MuData(adata_dict)
    mvatlas = MultiViewAtlas(mdata, transition_rule="louvain", keep_layers=False, keep_vars=False)
    assert mvatlas.mdata.mod["lymphoid"].n_vars == 0
    assert len(mvatlas.mdata.mod["lymphoid"].layers.keys()) == 0

    mdata2 = mudata.MuData(adata_dict)
    mvatlas = MultiViewAtlas(mdata2, transition_rule="louvain", keep_layers=True, keep_vars=True)
    assert "counts" in mvatlas.mdata.mod["lymphoid"].layers
    assert mvatlas.mdata.mod["myeloid"].n_vars > 0


def test_broken_assignment():
    adata = sample_dataset()
    # Broken assignment of cells to views
    adata.obsm["view_assign"]["T cells"] = adata.obsm["view_assign"]["T cells"].sample(frac=1).values
    with pytest.raises(AssertionError):
        MultiViewAtlas(adata)


def test_missing_view():
    adata = sample_dataset()
    # Missing view in hierarchy
    adata.obsm["view_assign"]["ciaone"] = adata.obsm["view_assign"]["T cells"].sample(frac=1).values
    with pytest.raises(AssertionError):
        MultiViewAtlas(adata)


def test_transition_rule():
    adata = sample_dataset()
    mva = MultiViewAtlas(adata)
    assert mva.view_transition_rule.shape[0] == mva.view_transition_rule.shape[1], "Transition rule is not square"
    child_v = "T cells"
    assert (
        mva.view_transition_rule.loc[get_parent_view(child_v, mva.view_hierarchy), child_v] is not None
    ), "Missing transition rule"
    assert mva.view_transition_rule.loc["full", child_v] is not None, "Extra transition rule"
    assert mva.view_transition_rule.loc["B cells", child_v] is not None, "Extra transition rule"


def test_update_views():
    adata = sample_dataset()
    mva = MultiViewAtlas(adata)
    transition_rule = "louvain"
    parent_view = "T cells"

    assign_tab = np.vstack(
        [
            np.where(mva.mdata[parent_view].obs[transition_rule] == "CD4 T cells", 1, 0),
            np.where(mva.mdata[parent_view].obs[transition_rule] == "CD8 T cells", 1, 0),
        ]
    ).T
    assign_tab = pd.DataFrame(assign_tab, columns=["CD4", "CD8"], index=mva.mdata[parent_view].obs_names)
    mva.update_views(parent_view="T cells", child_assign_tab=assign_tab, transition_rule=transition_rule)
    assert "CD4" in mva.views, "view not updated"
    assert "CD8" in mva.views, "view not updated"
    assert "CD8" in get_views_from_structure(mva.view_hierarchy), "view hierarchy not updated"
    assert "CD4" in mva.view_transition_rule.columns, "view transition rule not updated"
    assert "CD4" in mva.view_transition_rule.index, "view transition rule not updated"


def test_missing_full():
    """
    Test that init from anndata works even if full view is present just in one component
    """
    adata = sample_dataset()
    adata.obsm["view_assign"]
    adata.uns["view_hierarchy"] = {"full": adata.uns["view_hierarchy"]}

    mva = MultiViewAtlas(adata)
    assert "full" in mva.view_hierarchy.keys()
    assert "full" in mva.mdata.obsm["view_assign"].columns

    adata = sample_dataset()
    adata.obsm["view_assign"]["full"] = 1
    assert "full" in adata.obsm["view_assign"].columns and "full" not in adata.uns["view_hierarchy"].keys()

    mva = MultiViewAtlas(adata)
    assert "full" in mva.view_hierarchy.keys()
    assert "full" in mva.mdata.obsm["view_assign"].columns


def test_getter():
    adata = sample_dataset()
    adata.layers["counts"] = adata.X.copy()
    view_assign = adata.obsm["view_assign"].copy()
    adata_dict = {}
    adata_dict["full"] = adata.copy()
    for v in view_assign:
        adata_dict[v] = adata[view_assign[v] == 1].copy()
    for v in adata_dict.keys():
        sc.pp.normalize_total(adata_dict[v], target_sum=1e4)
        sc.pp.log1p(adata_dict[v])
        sc.pp.highly_variable_genes(adata_dict[v], n_top_genes=1000)

    mdata = mudata.MuData(adata_dict)
    mvatlas = MultiViewAtlas(mdata, transition_rule="louvain", keep_layers=False, keep_vars=True)
    diff_hvgs = np.setdiff1d(
        mvatlas.mdata["NKT cells"].var_names[mvatlas.mdata["NKT cells"].var["highly_variable"]],
        mvatlas.mdata["full"].var_names[mvatlas.mdata["full"].var["highly_variable"]],
    )
    assert len(diff_hvgs) > 0
    assert all(mvatlas["NKT cells"].var["highly_variable"] == mvatlas.mdata["NKT cells"].var["highly_variable"])
    diff_hvgs = np.setdiff1d(
        mvatlas["NKT cells"].var_names[mvatlas["NKT cells"].var["highly_variable"]],
        mvatlas.mdata["full"].var_names[mvatlas.mdata["full"].var["highly_variable"]],
    )
    assert len(diff_hvgs) > 0
