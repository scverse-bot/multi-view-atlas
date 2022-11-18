from typing import Union

import pandas as pd
import yaml
from anndata import AnnData
from mudata import MuData

from ..utils import get_parent_view, get_views_from_structure


class MultiViewAtlas:
    """Multi-view atlas methods"""

    def __init__(
        self,
        data: Union[AnnData, MuData] = None,
    ):
        """Initialize a MultiViewAltas object, encoding assignment to atlas views and hierarchy between views

        Params:
        --------
            data:
                MuData or AnnData
                if AnnData, must contain the following fields:
                - obsm["view_assign"]: binary DataFrame with assignment of each cells to views
                - uns["view_hierarchy"]: dictionary of hierarchy between views


            MuData: MuData object with original AnnData in `mudata['full']` and one modality for each dataset view.
            View AnnDatas only store obs and obsm.
        """
        if isinstance(data, AnnData):
            adata = data
            if "view_assign" not in adata.obsm.keys():
                raise ValueError("adata must contain assignment of cells to views in obsm['view_assign']")
            if "view_hierarchy" not in adata.uns.keys():
                raise ValueError("adata must contain dictionary of view hierarchy in uns['view_hierarchy']")

            _clean_view_assignment(adata)

            vdata_dict = {}
            vdata_dict["full"] = adata.copy()
            for v in adata.obsm["view_assign"].columns:
                vdata = adata[adata.obsm["view_assign"][v] == 1]
                vdata_dict[v] = AnnData(obs=vdata.obs, obsm=vdata.obsm)

            mdata = MuData(vdata_dict)
            mdata.uns["view_hierarchy"] = adata.uns["view_hierarchy"]
            mdata.obsm["view_assign"] = adata.uns["view_assign"]

        elif isinstance(data, MuData):
            mdata = data
            assert "full" in mdata.mod.keys()
            if "view_hierarchy" not in mdata.uns:
                try:
                    mdata.uns["view_hierarchy"] = mdata["full"].uns["view_hierarchy"]
                except KeyError:
                    raise ValueError("mdata must contain dictionary of view hierarchy in uns['view_hierarchy']")

            if "view_assign" not in mdata.obsm:
                try:
                    mdata.obsm["view_assign"] = mdata["full"].obsm["view_assign"]
                except KeyError:
                    view_assign = pd.DataFrame(index=mdata["full"].obs_names)
                    for k, v in mdata.mod.items():
                        if k != "full":
                            view_assign[k] = view_assign.index.isin(v.obs_names)
                    view_assign = view_assign.astype("int")
                    mdata["full"].obsm["view_assign"] = view_assign
                    _clean_view_assignment(mdata["full"])
                    mdata.obsm["view_assign"] = view_assign

        self.mdata = mdata

    def __getitem__(self, index) -> Union["MuData", AnnData]:
        if isinstance(index, str):
            try:
                vdata_full = self.mdata["full"][self.mdata[index].obs_names]
                vdata = AnnData(
                    # get attributes from the full view
                    X=vdata_full.X,
                    var=vdata_full.var,
                    varm=vdata_full.varm,
                    varp=vdata_full.varp,
                    # get attributes from the view
                    obs=self.mdata[index].obs,
                    obsm=self.mdata[index].obsm,
                    obsp=self.mdata[index].obsp,
                    uns=self.mdata[index].uns,
                )
                return vdata
            except KeyError:
                raise KeyError(f"View {index} not found")
        else:
            return self.mdata[index]

    def __repr__(self) -> str:
        l1 = "MultiViewAtlas object with view hierarchy:\n"
        hierarchy_str = yaml.dump(self.mdata.uns["view_hierarchy"])
        l2 = "\t" + "\n\t".join(hierarchy_str.split("\n"))
        l3 = self.mdata.__repr__()
        return l1 + l2 + "\n" + l3


def _clean_view_assignment(adata) -> None:
    """Check that the view assignment is correct"""
    assign_tab = adata.obsm["view_assign"]
    view_hierarchy = adata.uns["view_hierarchy"]

    # Check names match between dict and columns
    assign_cols_in_structure = assign_tab.columns.isin(get_views_from_structure(view_hierarchy))
    if not all(assign_cols_in_structure):
        missing_cols = assign_tab.columns[~assign_cols_in_structure].tolist()
        raise AssertionError(f"views {missing_cols} are missing in the view structure")

    # Check that all cells in child view are also in parent view
    for v in get_views_from_structure(view_hierarchy):
        parent_v = get_parent_view(v, view_hierarchy)
        if parent_v is not None:
            assert all(
                assign_tab[parent_v][assign_tab[v] == 1] == 1
            ), f"Not all cells in view {v} are in parent view {parent_v}"

    # Reorder from parents to children
    adata.obsm["view_assign"] = assign_tab[get_views_from_structure(view_hierarchy)].copy()
