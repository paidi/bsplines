import numpy as np
from bsplines import QuantileBSplineBasis
from bsplines.bsplines import compute_quantile_knots


def test_quartile_bspline():
    x = np.array(range(10))
    basis = QuantileBSplineBasis(0, 10, x, 3, 3)
    assert basis.design_matrix(x).shape == (10, 3)


def test_compute_quantile_knots():
    upper = 100
    x1 = np.array(range(upper))
    x2 = np.array(range(int(upper / 2)))
    n_bases = 8
    degree = 3
    k1 = compute_quantile_knots(0, upper, x1, n_bases, degree)
    k2 = compute_quantile_knots(0, upper, x2, n_bases, degree)
    np.testing.assert_array_equal(k1, k2)
