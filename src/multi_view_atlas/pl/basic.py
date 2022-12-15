from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn.objects as so
from mudata import MuData
from pandas.api.types import is_categorical_dtype
from scanpy.plotting._tools.scatterplots import _get_palette
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

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
    mvatlas: MultiViewAtlas,
    subset_obs: Union[List, None] = None,
    subsample_fraction: float = 0.1,
    text_offset: float = 0.1,
    figsize: List = (10, 4),
    pl_fontsize: int = 12,
    save: str = None,
    **kwargs,
):
    r"""Visualize view hierarchy and number of cells in each view

    Params:
    --------
        mvatlas:
            MultiViewAtlas object
        subset_obs:
            list of obs_names to show in plot
        subsample_fraction:
            fraction of cells to subsample in plot (for scalability, default=0.1)
        text_offset:
            offset for text labels on y axis (default: 0.1)
        figsize:
            figure size (default: (10, 4))
        pl_fontsize:
            fontsize for labels and axis labels (default: 12)
        save:
            path to save figure (default: None)
        \**kwargs:
            additional arguments to pass to `so.Plot.scale` (to customize appearance and palettes)

    Examples
    --------
    Plot view hierarchy and number of cells in each view

    >>> import multi_view_atlas as mva
    >>> adata = mva.utils.sample_dataset()
    >>> mvatlas = mva.MultiViewAtlas(adata)
    >>> mva.pl.view_hierarchy(mvatlas)

    Subset cells to plot

    >>> cells_of_interest = adata.obs_names[adata.obs["louvain"] == "CD4 T cells"]
    >>> mva.pl.view_hierarchy(mvatlas, fraction_subsample = 0.5, subset_obs=cells_of_interest)

    To speed up plotting while maintaining proportions, show a fraction of cells

    >>> mva.pl.view_hierarchy(mvatlas, fraction_subsample = 0.01)

    Change color palette

    >>> mva.pl.view_hierarchy(mvatlas, color="Spectral")
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

    # Pivot assignment to long format for plotting
    pl_df = view_assign.reset_index().melt(id_vars="index", var_name="view", value_name="value")
    pl_df = pl_df[pl_df["value"] == 1]  # Keep only cells assigned to view

    # Add view depth info
    view_depth_dict = dict(zip(all_views, view_depths))
    pl_df["view_depth"] = [-view_depth_dict[x] for x in pl_df.view]

    # Save number of cells per view
    n_cells_views = pl_df.groupby("view").size().to_dict()

    # Subsample cells for scalability
    if subsample_fraction < 1:
        sample_cells = np.random.choice(
            view_assign.index, size=int(np.round(subsample_fraction * view_assign.shape[0])), replace=False
        )
        pl_df = pl_df[pl_df["index"].isin(sample_cells)].copy()
        view_assign = view_assign.loc[sample_cells].copy()

    # Order cells by clustering
    # order_cells = view_assign.sort_values(all_views, ascending=False).index.tolist()
    Z = linkage(pdist(view_assign, metric="euclidean"), method="ward")
    order_cells = view_assign.index[dendrogram(Z, no_plot=True)["leaves"]]
    pl_df["index"] = pl_df["index"].astype("category")
    pl_df["index"].cat.reorder_categories(order_cells, inplace=True)
    pl_df["cell_order"] = pl_df["index"].cat.codes

    # Check for overlapping cells in the same depth
    for d in pl_df["view_depth"].unique():
        shared_cells = pl_df[pl_df["view_depth"] == d].value_counts("index")
        shared_cells = shared_cells.index[shared_cells > 1].tolist()
        if len(shared_cells) > 0:
            shared_views = pl_df.view[
                (pl_df["index"].isin(shared_cells)) & (pl_df["view_depth"] == d)
            ].unique()  # get views with shared cells
            # change depth to intermediate
            shared_depths = np.linspace(d + (0.4 / len(shared_views)), d - (0.4 / len(shared_views)), len(shared_views))
            for i, sv in enumerate(shared_views):
                pl_df.loc[pl_df.view == sv, "view_depth"] = shared_depths[i]

    # Make df for number of cells per view (for annotation)
    pl_df["n_cells"] = [n_cells_views[x] for x in pl_df.view]
    pl_df_ncells = pl_df.groupby(["n_cells", "view_depth", "view"]).median().reset_index()
    pl_df_ncells["label"] = [f"{v['view']} ({v['n_cells']} cells)" for _, v in pl_df_ncells.iterrows()]

    # Plot
    cells_offset = np.round(max(pl_df.cell_order) * 0.1)
    p = (
        so.Plot(pl_df, y="view_depth", x="cell_order", color="view", text="n_cells")
        .add(so.Dot(marker="|"))
        .scale(**kwargs)
        .limit(
            y=(min(pl_df.view_depth) - text_offset, max(pl_df.view_depth) + (text_offset * 5)),
            x=(min(pl_df.cell_order) - cells_offset, max(pl_df.cell_order) + (cells_offset)),
        )
        .label(x="Cells", y="View depth", color=None)
        .theme(
            {
                "axes.facecolor": "w",
                "axes.edgecolor": "C0",
                "xtick.bottom": False,
                "xtick.labelbottom": False,
                "ytick.left": False,
                "ytick.labelleft": False,
                "axes.labelsize": pl_fontsize,
            }
        )
    )

    f = matplotlib.figure.Figure(figsize=figsize)
    res = p.on(f).plot()
    ax = f.axes[0]
    for _, v in pl_df_ncells.iterrows():
        ax.text(
            x=v["cell_order"],
            y=v["view_depth"] + text_offset,
            s=v["label"],
            size=pl_fontsize,
            verticalalignment="bottom",
            horizontalalignment="center",
        )
    f.legends = []
    # remove frame from plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    if save is not None:
        print(f"Saving figure to {sc.settings.figdir / save}")
        res.save(sc.settings.figdir / save)

    return res
