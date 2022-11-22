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
    legend_loc: str = "on data",
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
        legend_loc:
            location of the legend
        fig_height:
            height of figure
        \**kwargs:
            additional arguments to pass to `sc.pl.embedding`
    """
    # Check if mdata is MultiViewAtlas
    if isinstance(mdata, MultiViewAtlas):
        mdata = mdata.mdata

    if view not in mdata.mod.keys():
        raise ValueError(f"View {view} not in mdata")

    # Get and order views to plot
    pl_views = []
    curr_view = view
    while curr_view is not None:
        pl_views.append(curr_view)
        curr_view = get_parent_view(curr_view, mdata.uns["view_hierarchy"])

    pl_views = pl_views[::-1]

    fig, ax = plt.subplots(1, len(pl_views), figsize=(fig_height * len(pl_views), fig_height))
    for i, v in enumerate(pl_views):
        if v == view:
            sc.pl.embedding(
                mdata[v],
                basis=basis,
                title=f"{v} view",
                color=color,
                legend_loc=legend_loc,
                ax=ax[i],
                show=False,
                **kwargs,
            )
        else:
            mdata.mod[v].obs["view_color"] = np.nan
            mdata.mod[v].obs.loc[mdata.mod[v].obs_names, "view_color"] = mdata.mod[v].obs.loc[
                mdata[view].obs_names, color
            ]
            if legend_loc != "on data":
                legend_loc_pl = "none"
            else:
                legend_loc_pl = legend_loc
            sc.pl.embedding(
                mdata[v],
                basis=basis,
                title=f"{v} view",
                color="view_color",
                legend_loc=legend_loc_pl,
                ax=ax[i],
                show=False,
                **kwargs,
            )

    plt.tight_layout(pad=3.0)
    plt.show()
