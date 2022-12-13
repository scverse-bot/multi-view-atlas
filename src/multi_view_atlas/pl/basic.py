from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import seaborn.objects as so
from mudata import MuData
from pandas.api.types import is_categorical_dtype
from scanpy.plotting._tools.scatterplots import _get_palette

from multi_view_atlas.tl import MultiViewAtlas
from multi_view_atlas.utils import get_parent_view


def multiview_embedding(
    mdata: Union[MultiViewAtlas, MuData],
    view: str,
    basis: str = "X_umap",
    basis_from_full: bool = False,
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
        basis_from_full:
            boolean indicating whether to use full view embedding (default: False, use view-specific embeddings)
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

    # # Make uniform color palette
    if is_categorical_dtype(mdata.mod[view].obs[color]):
        if "{color}_colors" not in mdata[view].uns.keys():
            mdata.mod[view].uns["{color}_colors"] = _get_palette(mdata[view], color).values()

    fig, ax = plt.subplots(1, len(pl_views), figsize=(fig_height * len(pl_views), fig_height))
    for i, v in enumerate(pl_views):
        # Define embedding basis
        if basis_from_full:
            if f"{basis}_full" in mdata["full"].obsm.keys():
                pl_basis = f"{basis}_full"
            elif basis in mdata["full"].obsm.keys():
                pl_basis = basis
            else:
                raise ValueError(f"Embedding {basis} not in mdata['full'].obsm")
            mdata.mod[v].obsm[pl_basis] = mdata.mod["full"][mdata.mod[v].obs_names].obsm[basis]
        else:
            pl_basis = f"{basis}_{v}"

        # Plot
        if v == view:
            sc.pl.embedding(
                mdata[v],
                basis=pl_basis,
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
            if is_categorical_dtype(mdata.mod[view].obs[color]):
                mdata.mod[v].obs["view_color"] = mdata.mod[v].obs["view_color"].astype("str").astype("category")
                mdata.mod[v].obs["view_color"] = np.where(
                    mdata.mod[v].obs["view_color"] == "nan", np.nan, mdata.mod[v].obs["view_color"]
                )
                mdata.mod[v].obs["view_color"] = mdata.mod[v].obs["view_color"].astype("category")
                mdata.mod[v].uns["view_color_colors"] = mdata.mod[view].uns["{color}_colors"]
            if legend_loc != "on data":
                legend_loc_pl = "none"
            else:
                legend_loc_pl = legend_loc
            sc.pl.embedding(
                mdata[v],
                basis=pl_basis,
                title=f"{v} view",
                color="view_color",
                legend_loc=legend_loc_pl,
                ax=ax[i],
                show=False,
                **kwargs,
            )

    plt.tight_layout(pad=3.0)
    plt.show()


def view_hierarchy(
    mvatlas: MultiViewAtlas, subset_obs: Union[List, None] = None, text_offset: float = 0.1, save: str = None, **kwargs
):
    r"""Visualize view hierarchy and number of cells in each view

    Params:
    --------
        mvatlas:
            MultiViewAtlas object
        subset_obs:
            list of obs_names to show in plot
        text_offset:
            offset for text labels on y axis (default: 0.1)
        save:
            path to save figure (default: None)
        \**kwargs:
            additional arguments to pass to `so.Plot.scale` (to customize appearance and palettes)

    """
    # Check input
    if not isinstance(mvatlas, MultiViewAtlas):
        raise ValueError("Input should be a MultiViewAtlas object")

    # Get view info
    all_views = ["full"] + mvatlas.get_view_pairs()["child_view"].tolist()
    view_depths = [0] + [d + 1 for d in mvatlas.get_view_pairs()["depth"]]

    # Get view assignment
    if subset_obs is None:
        view_assign = mvatlas.mdata.obsm["view_assign"].copy()
    else:
        view_assign = mvatlas.mdata.obsm["view_assign"].loc[subset_obs].copy()

    # Order cells by clustering
    cell_order_hm = sns.clustermap(view_assign[all_views], col_cluster=False)
    plt.close()
    ixs_cells = cell_order_hm.dendrogram_row.reordered_ind
    order_cells = view_assign.iloc[ixs_cells].index.tolist()

    # Pivot assignment to long format for plotting
    pl_df = view_assign.reset_index().melt(id_vars="index", var_name="view", value_name="value")
    pl_df["index"] = pl_df["index"].astype("category")
    pl_df["index"].cat.reorder_categories(order_cells, inplace=True)
    pl_df["cell_order"] = pl_df["index"].cat.codes
    pl_df = pl_df[pl_df["value"] == 1]  # Keep only cells assigned to view

    # Add view depth info
    view_depth_dict = dict(zip(all_views, view_depths))
    pl_df["view_depth"] = [-view_depth_dict[x] for x in pl_df.view]

    # Make df for number of cells per view (for annotation)
    n_cells_views = pl_df.groupby("view").size().to_dict()
    pl_df["n_cells"] = [n_cells_views[x] for x in pl_df.view]
    pl_df_ncells = pl_df.groupby(["n_cells", "view_depth", "view"]).median().reset_index()
    pl_df_ncells["label"] = [f"{v['view']} ({v['n_cells']} cells)" for _, v in pl_df_ncells.iterrows()]

    p = (
        so.Plot(pl_df, y="view_depth", x="cell_order", color="view", text="n_cells")
        .add(so.Dot(marker="s"))
        .scale(**kwargs)
        .limit(y=(min(pl_df.view_depth) - text_offset, max(pl_df.view_depth) + (text_offset * 5)))
        .label(x="cells", y="View depth", color=None)
        .theme(
            {
                "axes.facecolor": "w",
                "axes.edgecolor": "C0",
                "xtick.bottom": False,
                "xtick.labelbottom": False,
                "ytick.left": False,
                "ytick.labelleft": False,
                "axes.labelsize": 16,
            }
        )
    )

    f = matplotlib.figure.Figure(figsize=(10, 4))
    res = p.on(f).plot()
    ax = f.axes[0]
    for _, v in pl_df_ncells.iterrows():
        ax.text(
            x=v["cell_order"],
            y=v["view_depth"] + text_offset,
            s=v["label"],
            size=16,
            verticalalignment="bottom",
            horizontalalignment="center",
        )
    f.legends = []
    if save is not None:
        print(f"Saving figure to {sc.settings.figdir / save}")
        res.save(sc.settings.figdir / save)

    return res
