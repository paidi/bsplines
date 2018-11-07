import numpy as np

from scipy.interpolate import splev, BSpline


class BSplineBasis:
    def __init__(self, lower, upper, n_bases, degree):
        self.lower_bound = lower
        self.upper_bound = upper
        self.n_bases = n_bases
        self.degree = degree

        self._knots = compute_uniform_knots(lower, upper, n_bases, degree)
        self._basis = BSpline(self._knots, np.eye(n_bases), degree)

    def design_matrix(self, x):
        x = np.atleast_1d(x)
        cols = self._basis(x)
        return np.array(cols).T

    def __repr__(self):
        s = 'BSplineBasis(lower={}, upper={}, n_bases={}, degree={})'
        s = s.format(self.lower_bound, self.upper_bound, self.n_bases, self.degree)
        return s


class QuantileBSplineBasis:
    def __init__(self, lower, upper, x, n_bases, degree):
        self.lower_bound = lower
        self.upper_bound = upper
        self.n_bases = n_bases
        self.degree = degree

        self._knots = compute_quantile_knots(lower, upper, x, n_bases, degree)
        self._basis = BSpline(self._knots, np.eye(n_bases), degree)

    def design_matrix(self, x):
        x = np.atleast_1d(x)
        cols = self._basis(x)
        return np.array(cols).T

    def __repr__(self):
        s = 'QuantileBSplineBasis(lower={}, upper={}, n_bases={}, degree={})'
        s = s.format(self.lower_bound, self.upper_bound, self.n_bases, self.degree)
        return s


def compute_uniform_knots(lower, upper, n_bases, degree):
    n = n_interior_knots(n_bases, degree)
    k = np.linspace(lower, upper, n + 2)
    return pad_boundaries(k, degree)


def compute_quantile_knots(lower, upper, x, n_bases, degree):
    x = x[(lower <= x) & (x <= upper)]
    n = n_interior_knots(n_bases, degree)
    p = np.percentile(x, np.linspace(0, 100, n + 2))
    k = [lower] + list(p[1:-1]) + [upper]
    return pad_boundaries(k, degree)


def n_interior_knots(n_bases, degree):
    return n_bases - (degree + 1)


def pad_boundaries(knots, degree):
    knots = list(knots)
    padded = (knots[:1] * degree) + knots + (knots[-1:] * degree)
    return np.array(padded)
