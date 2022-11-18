import pytest

from multi_view_atlas.tl import MultiViewAtlas
from multi_view_atlas.utils import sample_dataset


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
