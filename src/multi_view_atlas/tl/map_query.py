import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import KNeighborsClassifier

from ..utils import check_transition_rule, get_views_from_structure
from .MultiViewAtlas import MultiViewAtlas

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def add_query(
    mvatlas: MultiViewAtlas,
    adata_query: AnnData,
) -> MultiViewAtlas:
    """Map query data to the multi-view atlas

    Params:
    -------
        mvatlas:
            MultiViewAtlas object
        adata_query:
            AnnData of query data

    Returns:
    --------
        MultiViewAtlas object with mapped query cells
    """
    # Define all current view - next view pairs for assignment
    view_pairs = []
    view_str = pd.json_normalize(mvatlas.view_hierarchy).columns.str.split(".")
    for s in view_str:
        depth = 0
        while depth < (len(s) - 1):
            # view_pair = s[depth: depth + 2]
            view_pairs.append((depth, s[depth], s[depth + 1]))
            depth += 1
    view_pairs = pd.DataFrame(view_pairs, columns=["depth", "current_view", "next_view"])
    view_pairs = view_pairs.sort_values("depth")
    view_pairs = view_pairs.drop_duplicates()

    # Check if query cells already in mdata
    if adata_query.obs_names.isin(mvatlas.mdata.obs_names).all():
        vdata_full = mvatlas.mdata["full"].copy()
    else:
        vdata_full = concatenate_query(mvatlas, adata_query, "full")

    # Check if query cells already in mdata
    if adata_query.obs_names.isin(mvatlas.mdata.obs_names).all():
        vdata_full = mvatlas.mdata["full"].copy()
    else:
        vdata_full = concatenate_query(mvatlas, adata_query, "full")

    new_view_assign = pd.DataFrame()
    vdata_dict = {}
    for _, row in view_pairs.iterrows():
        depth = row["depth"]
        current_view = row["current_view"]
        next_view = row["next_view"]
        if any(adata_query.obs_names.isin(mvatlas.mdata[next_view].obs_names)):
            logging.info(f"Query cells already in {next_view}")
            v_assign = mvatlas.mdata.obsm["view_assign"][[next_view]]
            vdata_dict[next_view] = mvatlas.mdata[next_view].copy()
        else:
            print(f"Assigning to {next_view} from {current_view}")
            adata_query_concat = AnnData(obs=adata_query.obs, obsm=adata_query.obsm, obsp=adata_query.obsp)
            if depth > 0:
                adata_query_concat = adata_query_concat[
                    new_view_assign.loc[adata_query_concat.obs_names, current_view] == 1
                ].copy()
            v_assign, next_view_adata = map_next_view(
                mvatlas, adata_query_concat, current_view=current_view, next_view=next_view
            )
            vdata_dict[next_view] = next_view_adata
        new_view_assign = pd.concat([new_view_assign, v_assign], 1)

    new_view_assign = new_view_assign.fillna(0)
    vdata_full.obsm["view_assign"] = new_view_assign.copy()
    vdata_full.uns["view_hierarchy"] = mvatlas.view_hierarchy
    vdata_dict["full"] = vdata_full
    mdata = MuData({v: vdata_dict[v] for v in get_views_from_structure(mvatlas.view_hierarchy)})
    mdata.mod["full"] = mdata.mod["full"][mdata.obs_names].copy()
    view_transition_rule = mvatlas.view_transition_rule.copy()
    trans_rule = pd.Series(mvatlas.view_transition_rule.values.ravel()).dropna().unique()[0]
    mvatlas_mapped = MultiViewAtlas(mdata, transition_rule=trans_rule)
    mvatlas_mapped.view_transition_rule = view_transition_rule.copy()
    return mvatlas_mapped


def concatenate_query(
    mvatlas: MultiViewAtlas,
    adata_query: AnnData,
    view: str,
    batch_key: str = "dataset_group",
    batch_categories: List[str] = None,
):
    """Concatenate query data to atlas data"""
    if batch_categories is None:
        batch_categories = ["atlas", "query"]

    adata_atlas = mvatlas.mdata[view].copy()
    assert np.intersect1d(adata_atlas.obs_names, adata_query.obs_names).size == 0
    adata_atlas = adata_atlas.concatenate(
        adata_query, batch_key=batch_key, batch_categories=batch_categories, index_unique=None, uns_merge="unique"
    )
    return adata_atlas


def assign_from_similarity(Y_assign_atlas, X_similarity_atlas, X_similarity_query, v_assign, k=10):
    """Assign query cells to atlas views based on similarity to atlas cells.

    Similarity can be derived from metadata annotation or from latent space
    """
    if not isinstance(v_assign, pd.DataFrame):
        raise ValueError("v_assign must be a pandas DataFrame")
    if not v_assign.shape[1] == 1:
        raise ValueError("v_assign must have only one column")

    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(X_similarity_atlas, Y_assign_atlas)
    Y_assign_query = clf.predict(X_similarity_query)
    assign_predict = pd.DataFrame(Y_assign_query, columns=v_assign.columns)
    return assign_predict


def map_next_view(
    mvatlas: MultiViewAtlas,
    adata_query: AnnData,
    current_view: str,
    next_view: str,
    batch_key: str = "dataset_group",
    batch_categories: List[str] = None,
) -> Tuple[pd.DataFrame, AnnData]:
    """Assign query cells to next view based on similarity to atlas cells.

    Similarity is defined by the transition rule.
    """
    if batch_categories is None:
        batch_categories = ["atlas", "query"]

    # curr_view_adata = _concatenate_query(mvatlas, adata_query, current_view, batch_key=batch_key, batch_categories=batch_categories)
    curr_view_adata = mvatlas.mdata[current_view].copy()
    next_view_adata = mvatlas.mdata[next_view].copy()

    # Get similarity from transition rule
    if batch_key in curr_view_adata.obs.columns:
        v_assign = mvatlas.mdata.obsm["view_assign"].loc[mvatlas.mdata["full"].obs[batch_key] == batch_categories[0]][
            [next_view]
        ]
        curr_view_adata = curr_view_adata[curr_view_adata.obs[batch_key] == batch_categories[0]].copy()
        # next_view_adata = next_view_adata[next_view_adata.obs[batch_key] == batch_categories[0]].copy()
        # assert "dataset_group" not in next_view_adata.obs.columns
    else:
        v_assign = mvatlas.mdata.obsm["view_assign"][[next_view]]
    transition_rule = mvatlas.view_transition_rule[current_view][next_view]
    print(f"Transition rule: {transition_rule}")
    if transition_rule is not None:
        try:
            check_transition_rule(adata_query, transition_rule)
        except ValueError:
            logging.warning(
                f"Could not check transition rule {transition_rule} for query data - skipping mapping from {current_view} to {next_view}"
            )
            v_assign_query = pd.DataFrame(0, columns=[next_view], index=adata_query.obs_names)
            v_assign = pd.concat([v_assign, v_assign_query], axis=0)
            return v_assign, next_view_adata
        # Split to next view based on transition rule
        # if transition is in obs
        if transition_rule in adata_query.obs.columns:
            if is_numeric_dtype(adata_query.obs[transition_rule]):
                X_similarity_atlas = curr_view_adata.obs[[transition_rule]].values
                X_similarity_query = adata_query.obs[[transition_rule]].values
            else:
                X_similarity_atlas = pd.get_dummies(curr_view_adata.obs[transition_rule])
                X_similarity_query = pd.get_dummies(adata_query.obs[transition_rule])

                # Check for missing levels
                missing_cols = [x for x in X_similarity_atlas.columns if x not in X_similarity_query.columns]
                if len(missing_cols) > 0:
                    X_similarity_query[missing_cols] = 0
                    X_similarity_query = X_similarity_query[X_similarity_atlas.columns].copy()

                X_similarity_atlas = X_similarity_atlas.values
                X_similarity_query = X_similarity_query.values

        if transition_rule in adata_query.obsm:
            X_similarity_atlas = curr_view_adata.obsm[transition_rule]
            X_similarity_query = adata_query.obsm[transition_rule]
        # X_similarity = curr_view_adata.obsm[transition_rule]
    else:
        raise ValueError(f"No transition rule defined for {current_view} -> {next_view}")

    # Get assignment to next view
    Y_assign_atlas = v_assign.loc[curr_view_adata.obs_names].values.ravel()

    v_assign_query = assign_from_similarity(Y_assign_atlas, X_similarity_atlas, X_similarity_query, v_assign)
    v_assign_query.index = adata_query.obs_names

    v_assign = pd.concat([v_assign, v_assign_query], axis=0)
    next_view_adata = concatenate_query(
        mvatlas,
        adata_query[v_assign_query[next_view] == 1],
        next_view,
        batch_key=batch_key,
        batch_categories=batch_categories,
    )

    return v_assign, next_view_adata
