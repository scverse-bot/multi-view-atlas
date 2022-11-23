import mudata
import numpy as np
import pandas as pd
import pytest

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
