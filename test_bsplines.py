import numpy as np
from bsplines import QuantileBSplineBasis


def test_quartile_bspline():
    x = np.array(range(10))
    basis = QuantileBSplineBasis(0, 10, x, 3, 3)
    assert basis.design_matrix(x).shape == (10, 3)