from typing import Dict, Union

import numpy as np
import pandas as pd


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
        if v in s:
            view_hierarchy = np.array(s.split("."))
            parent_ix = [i - 1 for i, v1 in enumerate(view_hierarchy) if v == v1][0]
    if parent_ix == -1:
        parent_view = None
    else:
        parent_view = view_hierarchy[parent_ix]
    return parent_view
