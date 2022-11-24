from typing import List, Union

import numpy as np
import pandas as pd
import yaml
from anndata import AnnData
from mudata import MuData

from ..utils import check_transition_rule, get_parent_view, get_views_from_structure


class MultiViewAtlas:
    """Multi-view atlas class"""

    def __init__(
        self,
        data: Union[AnnData, MuData] = None,
        transition_rule: Union[str, List[str], pd.DataFrame] = None,
        subset_obsm: bool = False,
        rename_obsm: bool = True,
    ):
        """Initialize a MultiViewAltas object, encoding assignment to atlas views and hierarchy between views

        Params:
        --------
            data:
                MuData or AnnData
                if AnnData, must contain the following fields:
                - obsm["view_assign"]: binary DataFrame with assignment of each cells to views
                - uns["view_hierarchy"]: dictionary of hierarchy between views
            transition_rule: str or list
                which rule to use for transition between one view and another: either a slot in adata.obsm storing latent dimensions (i.e. transition by clustering)
                or a column in adata.obs or list of columns (i.e. transition by metadata)
            subset_obsm: bool
                (used only if data is an AnnData object) whether to store a subset of the full data obsm
                in every view (default: False, obsm slots are considered to be specific to the full view)
            rename_obsm: bool
                (used only if data is an MuData object) whether to rename obsm slots to avoid name clashes

            MuData: MuData object with original AnnData in `mudata['full']` and one modality for each dataset view.
            View AnnDatas only store obs and obsm.

        Details:
        --------
            All the views in `view_assign` have to be in `'view_hierarchy`, but not viceversa
            to allow building the hierarchy progressively.
        """
        if isinstance(data, AnnData):
            adata = data
            if "view_assign" not in adata.obsm.keys():
                raise ValueError("adata must contain assignment of cells to views in obsm['view_assign']")
            if "view_hierarchy" not in adata.uns.keys():
                raise ValueError("adata must contain dictionary of view hierarchy in uns['view_hierarchy']")

            if "full" not in adata.uns["view_hierarchy"].keys():
                adata.uns["view_hierarchy"] = {"full": adata.uns["view_hierarchy"]}
            if "full" not in adata.obsm["view_assign"]:
                adata.obsm["view_assign"]["full"] = 1

            vdata_dict = {}
            vdata_dict["full"] = adata.copy()
            for v in adata.obsm["view_assign"].columns:
                if v != "full":
                    vdata = adata[adata.obsm["view_assign"][v] == 1]
                    if subset_obsm:
                        vdata_dict[v] = AnnData(obs=vdata.obs, obsm=vdata.obsm, obsp=vdata.obsp)
                    else:
                        vdata_dict[v] = AnnData(obs=vdata.obs)

            _clean_view_assignment(adata)

            mdata = MuData(vdata_dict)
            mdata.obs = mdata["full"].obs.copy()

            mdata.uns["view_hierarchy"] = adata.uns["view_hierarchy"]
            mdata.obsm["view_assign"] = adata.obsm["view_assign"]

        elif isinstance(data, MuData):
            mdata = data
            assert "full" in mdata.mod.keys()
            if "view_hierarchy" not in mdata.uns:
                try:
                    mdata.uns["view_hierarchy"] = mdata["full"].uns["view_hierarchy"]
                except KeyError:
                    raise ValueError("mdata must contain dictionary of view hierarchy in uns['view_hierarchy']")

            if "full" not in mdata.uns["view_hierarchy"].keys():
                mdata.uns["view_hierarchy"] = {"full": mdata.uns["view_hierarchy"]}

            if "view_assign" not in mdata.obsm:
                try:
                    mdata.obsm["view_assign"] = mdata["full"].obsm["view_assign"]
                except KeyError:
                    view_assign = pd.DataFrame(index=mdata["full"].obs_names)
                    for k, v in mdata.mod.items():
                        view_assign[k] = view_assign.index.isin(v.obs_names)
                    view_assign = view_assign.astype("int")
                    mdata["full"].obsm["view_assign"] = view_assign
                    _clean_view_assignment(mdata["full"])
                    mdata.obsm["view_assign"] = view_assign

            # Remove var and X from views
            for k in mdata.mod.keys():
                if k != "full":
                    mdata.mod[k] = AnnData(obs=mdata[k].obs, obsm=mdata[k].obsm)

            # Rename obsm slots to be view specific
            if rename_obsm:
                for v in mdata.mod.keys():
                    obsm_dict = mdata.mod[v].obsm.copy()
                    obsm_dict = {f"{k}_{v}": dr for k, dr in obsm_dict.items()}
                    mdata.mod[v].obsm = obsm_dict.copy()

        else:
            raise ValueError("data must be an AnnData or MuData object")

        # Make matrix of transition rules
        if transition_rule is not None:
            if isinstance(transition_rule, str) or isinstance(transition_rule, list):
                check_transition_rule(mdata["full"], transition_rule)
                view_hierarchy = mdata.uns["view_hierarchy"].copy()
                all_views = get_views_from_structure(view_hierarchy)
                view_transition_rule = pd.DataFrame(np.nan, index=all_views, columns=all_views)
                # Set all transitions to rule
                view_str = pd.json_normalize(view_hierarchy).columns.tolist()
                for s in view_str:
                    v_str = np.array(s.split("."))
                    for v in v_str:
                        view_transition_rule.loc[v, v_str] = transition_rule
                np.fill_diagonal(view_transition_rule.values, np.nan)
            elif isinstance(transition_rule, pd.DataFrame):
                view_transition_rule = transition_rule.copy()
        else:
            view_hierarchy = mdata.uns["view_hierarchy"].copy()
            all_views = get_views_from_structure(view_hierarchy)
            view_transition_rule = pd.DataFrame(np.nan, index=all_views, columns=all_views)

        self.mdata = mdata
        self.views = get_views_from_structure(self.mdata.uns["view_hierarchy"])
        self.view_hierarchy = self.mdata.uns["view_hierarchy"]
        self.view_transition_rule = view_transition_rule

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

    # def copy(self) -> "MultiViewAtlas":
    #     return MultiViewAtlas(self.mdata.copy())

    def set_transition_rule(self, parent_view, child_view, transition_rule):
        """Set new transition rule between two views.

        Parameters
        ----------
        parent_view: str
            name of parent view
        child_view: str
            name of child view
        transition_rule: str or list
            which rule to use for transition between one view and another: either a slot in adata.obsm storing latent dimensions (i.e. transition by clustering)
            or a column in adata.obs or list of columns (i.e. transition by metadata)

        Returns
        -------
        None, modifies MultiViewAtlas object in place

        """
        if not get_parent_view(child_view, self.mdata.uns["view_hierarchy"]) == parent_view:
            raise AssertionError(f"View {child_view} is not a child of {parent_view}")

        check_transition_rule(self.mdata[parent_view], transition_rule)
        self.view_transition_rule.loc[
            [parent_view, child_view], [parent_view, child_view]
        ] = self.view_transition_rule.loc[[parent_view, child_view], [parent_view, child_view]].apply(
            lambda _: transition_rule, axis=1
        )
        np.fill_diagonal(self.view_transition_rule.values, np.nan)

    def update_views(self, parent_view, child_assign_tab=None, transition_rule=None):
        """Add new views in the MultiViewAtlas object.

        Parameters
        ----------
        parent_view: str
            name of parent view to subset from
        child_assign_tab: pd.DataFrame
            table of view assignments for each cell in the parent view
        transition_rule: str or list
            which rule to use for transition between one view and another: either a slot in adata.obsm storing latent dimensions (i.e. transition by clustering)
            or a column in adata.obs or list of columns (i.e. transition by metadata)

        Returns
        -------
        None, modifies MultiViewAtlas object in place adding new views

        """
        child_views = child_assign_tab.columns.tolist()

        # Update view assignment
        self.mdata.obsm["view_assign"][child_views] = child_assign_tab.astype(int)
        self.mdata.obsm["view_assign"].fillna(0, inplace=True)
        self.mdata.obsm["view_assign"] = self.mdata.obsm["view_assign"].astype(int)

        # Update view_hierarchy
        v_keys = [x for x in pd.json_normalize(self.view_hierarchy).columns.tolist() if x.endswith(parent_view)][
            0
        ].split(".")
        self.view_hierarchy = _dict_set_nested(self.view_hierarchy, v_keys, {v: None for v in child_views})

        _clean_view_assignment(self.mdata)

        # Add AnnData objects for new views
        for v in child_views:
            vdata = self.mdata["full"][self.mdata.obsm["view_assign"][v] == 1]
            self.mdata.mod[v] = AnnData(obs=vdata.obs, obsm=vdata.obsm, obsp=vdata.obsp)

        # Update transition_rule
        check_transition_rule(self.mdata, transition_rule)
        self.view_transition_rule[[parent_view] + child_views] = np.nan
        view_transition_rule = pd.DataFrame(np.nan, index=child_views, columns=[parent_view] + child_views)
        self.view_transition_rule = pd.concat([self.view_transition_rule, view_transition_rule], axis=0)
        for v in child_views:
            self.view_transition_rule.loc[[parent_view, v], [parent_view, v]] = self.view_transition_rule.loc[
                [parent_view, v], [parent_view, v]
            ].apply(lambda _: transition_rule, axis=1)
        np.fill_diagonal(self.view_transition_rule.values, np.nan)
        self.views = get_views_from_structure(self.mdata.uns["view_hierarchy"])

    def get_view_pairs(self) -> pd.DataFrame:
        """Get data frame of parent-child view pairs and transition rules."""
        view_pairs = []
        view_str = pd.json_normalize(self.view_hierarchy).columns.str.split(".")
        for s in view_str:
            depth = 0
            while depth < (len(s) - 1):
                # view_pair = s[depth: depth + 2]
                view_pairs.append((depth, s[depth], s[depth + 1]))
                depth += 1
        view_pairs = pd.DataFrame(view_pairs, columns=["depth", "parent_view", "child_view"])
        view_pairs["transition_rule"] = [
            self.view_transition_rule.loc[x[1]["parent_view"], x[1]["child_view"]] for x in view_pairs.iterrows()
        ]
        view_pairs = view_pairs.sort_values("depth")
        view_pairs = view_pairs.drop_duplicates()
        return view_pairs


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


def _dict_set_nested(d, keys, value):
    node = d
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return d
        else:
            if key not in node:
                node[key] = {}
                node = node[key]
            else:
                node = node[key]
