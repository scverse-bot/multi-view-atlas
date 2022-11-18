from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from mudata import MuData

from multi_view_atlas.tl import MultiViewAtlas
from multi_view_atlas.utils import get_parent_view


def multiview_embedding(
    mdata: Union[MultiViewAtlas, MuData],
    view: str,
    basis: str = "X_umap",
    color: str = "louvain",
    fig_height: float = 6,
    **kwargs,
):
    r"""Plot embedding showing multiple views

    Params:
    --------
        mdata:
            MultiViewAtlas or MuData object
        view:
            view to plot
        basis:
            embedding to plot (slot in `mdata[view].obsm`)
        color:
            color to plot (slot in `mdata[view].obs`)
        fig_height:
            height of figure
        \**kwargs:
            additional arguments to pass to `sc.pl.embedding`
    """
    # Get and order views to plot
    pl_views = []
    curr_view = view
    while curr_view is not None:
        pl_views.append(curr_view)
        curr_view = get_parent_view(curr_view, mdata["full"].uns["view_hierarchy"])

    pl_views.append("full")
    pl_views = pl_views[::-1]

    fig, ax = plt.subplots(1, len(pl_views), figsize=(fig_height * len(pl_views), fig_height))
    for i, v in enumerate(pl_views):
        if v == pl_views[-1]:
            sc.pl.embedding(mdata[v], basis=basis, title=f"{v} view", color="louvain", ax=ax[i], show=False, **kwargs)
        else:
            adata_pl = mdata[v].copy()
            adata_pl.obs["view_color"] = np.nan
            adata_pl.obs.loc[mdata[view].obs_names, "view_color"] = adata_pl.obs.loc[mdata[view].obs_names, color]
            sc.pl.embedding(
                adata_pl,
                basis=basis,
                title=f"{v} view",
                color="view_color",
                legend_loc=None,
                ax=ax[i],
                show=False,
                **kwargs,
            )

    plt.tight_layout(pad=3.0)
    plt.show()
