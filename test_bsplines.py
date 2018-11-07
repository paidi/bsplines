import numpy as np
import pytest

from bsplines import QuantileBSplineBasis


def test_quartile_bspline():
    x = np.array(range(10))
    basis = QuantileBSplineBasis(0, 10, x, 4, 3)
    with pytest.warns(None) as record:
        assert basis.design_matrix(x).shape == (10, 4)
    assert not record.list
