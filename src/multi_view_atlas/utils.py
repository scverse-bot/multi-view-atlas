from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scanpy as sc


def get_views_from_structure(view_hierarchy: Dict):
    """Get list of all views from view hierarchy"""

    def _recursive_items(dictionary):
        for key, value in dictionary.items():
            if type(value) is dict:
                yield (key, value)
                yield from _recursive_items(value)
            else:
                yield (key, value)

    return [k for k, v in _recursive_items(view_hierarchy)]


def get_parent_view(v, view_hierarchy: Dict) -> Union[str, None]:
    """Get parent view of view v"""
    view_str = pd.json_normalize(view_hierarchy).columns.tolist()
    for s in view_str:
        if v in np.array(s.split(".")):
            view_hierarchy_ls = np.array(s.split("."))
            parent_ix = [i - 1 for i, v1 in enumerate(view_hierarchy_ls) if v == v1][0]
    if parent_ix == -1:
        parent_view = None
    else:
        parent_view = view_hierarchy_ls[parent_ix]
    return parent_view


def sample_dataset():
    """Example dataset for testing"""
    adata = sc.datasets.pbmc3k_processed()
    adata = adata.raw.to_adata()
    # adata2 = sc.datasets.blobs(n_observations=10000, n_centers=12, n_variables=500)
    # Make DataFrame assigning cells to views
    assign_dict = {
        "myeloid": ["CD14+ Monocytes", "FCGR3A+ Monocytes", "Dendritic cells", "Megakaryocytes"],
        "lymphoid": ["NK cells", "CD8 T cells", "CD4 T cells", "B cells"],
        "NKT cells": ["NK cells", "CD8 T cells", "CD4 T cells"],
        "T cells": ["CD8 T cells", "CD4 T cells"],
        "B cells": ["B cells"],
    }
    annotation_col = "louvain"

    assign_tab = np.vstack(
        [np.where(adata.obs[annotation_col].isin(assign_dict[k]), 1, 0) for k in assign_dict.keys()]
    ).T
    assign_tab = pd.DataFrame(assign_tab, columns=assign_dict.keys(), index=adata.obs_names)

    #  Make dictionary of parent-child structure of views
    view_hierarchy = {"myeloid": None, "lymphoid": {"NKT cells": {"T cells": None}, "B cells": None}}
    adata.obsm["view_assign"] = assign_tab.copy()
    adata.uns["view_hierarchy"] = view_hierarchy.copy()
    return adata


def check_transition_rule(adata, transition_rule):
    """Check that transition rule is of acceptable type"""
    if isinstance(transition_rule, str):
        if transition_rule in adata.obsm.keys():
            pass
        elif transition_rule in adata.obs:
            pass
        else:
            raise ValueError(f"transition_rule {transition_rule} not found in .obsm or .obs")
    elif isinstance(transition_rule, List):
        if not all([tr in adata.obs for tr in transition_rule]):
            raise ValueError(f"{transition_rule} are not found in .obs")
    else:
        raise ValueError("transition_rule must be a string or a list of strings")
