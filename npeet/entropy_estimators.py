#!/usr/bin/env python
# Written by Greg Ver Steeg
# See readme.pdf for documentation
# Or go to http://www.isi.edu/~gregv/npeet.html

import warnings

import numpy as np
import numpy.linalg as la
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree
from numba import njit

# CONTINUOUS ESTIMATORS


def entropy(x, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def centropy(x, y, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    entropy_union_xy = entropy(xy, k=k, base=base)
    entropy_y = entropy(y, k=k, base=base)
    return entropy_union_xy - entropy_y


def tc(xs, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropy(col, k=k, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropy(xs, k, base)


def ctc(xs, y, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropy(col, y, k=k, base=base)
                         for col in xs_columns]
    return np.sum(centropy_features) - centropy(xs, y, k, base)


def corex(xs, ys, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [mi(col, ys, k=k, base=base) for col in xs_columns]
    return np.sum(cmi_features) - mi(xs, ys, k=k, base=base)


def mi(x, y, z=None, k=3, base=2, alpha=0):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    n = len(x)
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(n, -1), y.reshape(n, -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    dx = x.shape[-1]
    dy = y.shape[-1]
    if z is not None:
        assert len(z) == n, "Arrays should have same length"
        z = np.asarray(z)
        z = z.reshape(n, -1)
        points.append(z)
        dz = z.shape[-1]
    else:
        dz = 0
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(
            y, dvec), digamma(k), digamma(len(x))
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(
            yz, dvec), avgdigamma(z, dvec), digamma(k)
    if alpha is None:
        alpha = alpha_estimation(k, dx, dy, dz=dz)
    if alpha > 0:
        d += lnc_correction(k, alpha, tree, points, x, y, z=z)
    return (-a - b + c + d) / log(base)


def mi2(x, y, z=None, k=3, base=2, alpha=0):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    n = len(x)
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(n, -1), y.reshape(n, -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    dx = x.shape[-1]
    dy = y.shape[-1]
    if z is not None:
        assert len(z) == n, "Arrays should have same length"
        z = np.asarray(z)
        z = z.reshape(n, -1)
        points.append(z)
        dz = z.shape[-1]
    else:
        dz = 0
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    idx = query_neighbors_index(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma2(x, idx), avgdigamma2(
            y, idx), digamma(k) - 1/k, digamma(len(x))
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma2(xz, idx), avgdigamma2(
            yz, idx), avgdigamma2(z, idx), digamma(k) - 1/k
    if alpha is None:
        alpha = alpha_estimation(k, dx, dy, dz=dz)
    if alpha > 0:
        d += lnc_correction(k, alpha, tree, points, x, y, z=z)
    return (-a - b + c + d) / log(base)


def cmi(x, y, z, k=3, base=2):
    """ Mutual information of x and y, conditioned on z
        Legacy function. Use mi(x, y, z) directly.
    """
    return mi(x, y, z=z, k=k, base=base)


def kldiv(x, xp, k=3, base=2):
    """ KL Divergence between p and q for x~p(x), xp~q(x)
        x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k < min(len(x), len(xp)), "Set k smaller than num. samples - 1"
    assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
    x, xp = np.asarray(x), np.asarray(xp)
    x, xp = x.reshape(x.shape[0], -1), xp.reshape(xp.shape[0], -1)
    d = len(x[0])
    n = len(x)
    m = len(xp)
    const = log(m) - log(n - 1)
    tree = build_tree(x)
    treep = build_tree(xp)
    nn = query_neighbors(tree, x, k)
    nnp = query_neighbors(treep, x, k - 1)
    return (const + d * (np.log(nnp).mean() - np.log(nn).mean())) / log(base)


def lnc_correction(k, alpha, tree, points, x, y, z=None):
    e = 0
    n_sample = points.shape[0]

    if z is not None:  # or give xz, yz instead of x, y ?
        x = np.hstack([x, z])
        y = np.hstack([y, z])

    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]

        knn_points_xyz = points[knn]
        knn_points_x = x[knn]
        knn_points_y = y[knn]

        log_V_rect_xyz = pca_volume(knn_points_xyz)
        log_V_rect_x = pca_volume(knn_points_x)
        log_V_rect_y = pca_volume(knn_points_y)

        if z is not None:
            knn_points_z = z[knn]
            log_V_rect_z = pca_volume(knn_points_z)
        else:
            log_V_rect_z = 0

        log_ratio = log_V_rect_xyz + log_V_rect_z - \
            (log_V_rect_x + log_V_rect_y)
        # Perform local non-uniformity checking and update correction term
        if log_ratio < np.log(alpha):
            e += - log_ratio
    return e / n_sample


@njit
def alpha_estimation(k, dx, dy, dz=0, N=int(5e5), eps=5e-3):
    d = dx + dy + dz

    a = np.empty(N)
    points = np.random.rand(N, k, d)

    if dz != 0:
        x_sel = np.array([i for i in range(dx)] + [i for i in range(dx+dy, d)])

    for i in range(N):
        knn_points_xyz = points[i, :, :]
        knn_points_x = points[i, :, :dx]  # -dz:dx not conitguous
        knn_points_y = points[i, :, dx:]

        if dz != 0:
            knn_points_z = points[i, :, -dz:]
            knn_points_x = points[i][:, x_sel]

            log_V_rect_z = pca_volume(knn_points_z)
        else:
            log_V_rect_z = 0

        log_V_rect_xyz = pca_volume(knn_points_xyz)
        log_V_rect_x = pca_volume(knn_points_x)
        log_V_rect_y = pca_volume(knn_points_y)

        log_ratio = log_V_rect_xyz + log_V_rect_z - \
                    (log_V_rect_x + log_V_rect_y)
        a[i] = log_ratio

    a = np.exp(a)
    choice = int(np.ceil(eps*N))
    return np.partition(a, choice - 1)[choice - 1]


# DISCRETE ESTIMATORS
def entropyd(sx, base=2):
    """ Discrete entropy estimator
        sx is a list of samples
    """
    unique, count = np.unique(sx, return_counts=True, axis=0)
    # Convert to float as otherwise integer division results in all 0 for proba.
    proba = count.astype(float) / len(sx)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / log(base)


def entropyd_jk(sx, k=10, base=2):
    """ Discrete entropy estimator
        sx is a list of samples

        Jack-knifed with k-folding (k=len(x) for maximum accuracy)
    """
    N = len(sx)
    r = N / k
    h = entropyd(sx, base=base)
    kfold_h = np.mean([entropyd(np.delete(sx, slice(int(r * j), int(r * (j+1))), axis=0),
                                base=base) for j in range(k)])
    return k * h - (k - 1) * kfold_h


def midd(x, y, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    assert len(x) == len(y), "Arrays should have same length"
    return entropyd(x, base) - centropyd(x, y, base)


def cmidd(x, y, z, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    assert len(x) == len(y) == len(z), "Arrays should have same length"
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    return entropyd(xz, base) + entropyd(yz, base) - entropyd(xyz, base) - entropyd(z, base)


def centropyd(x, y, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    return entropyd(xy, base) - entropyd(y, base)


def tcd(xs, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropyd(col, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropyd(xs, base)


def ctcd(xs, y, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropyd(col, y, base=base) for col in xs_columns]
    return np.sum(centropy_features) - centropyd(xs, y, base)


def corexd(xs, ys, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [midd(col, ys, base=base) for col in xs_columns]
    return np.sum(cmi_features) - midd(xs, ys, base)


# MIXED ESTIMATORS
def micd(x, y, k=3, base=2, warning=True):
    """ If x is continuous and y is discrete, compute mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    entropy_x = entropy(x, k, base)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * entropy(x_given_y, k, base)
        else:
            if warning:
                warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                              "Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    return abs(entropy_x - entropy_x_given_y)  # units already applied


def midc(x, y, k=3, base=2, warning=True):
    return micd(y, x, k, base, warning)


def centropycd(x, y, k=3, base=2, warning=True):
    return entropy(x, base) - micd(x, y, k, base, warning)


def centropydc(x, y, k=3, base=2, warning=True):
    return centropycd(y, x, k=k, base=base, warning=warning)


def ctcdc(xs, y, k=3, base=2, warning=True):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropydc(
        col, y, k=k, base=base, warning=warning) for col in xs_columns]
    return np.sum(centropy_features) - centropydc(xs, y, k, base, warning)


def ctccd(xs, y, k=3, base=2, warning=True):
    return ctcdc(y, xs, k=k, base=base, warning=warning)


def corexcd(xs, ys, k=3, base=2, warning=True):
    return corexdc(ys, xs, k=k, base=base, warning=warning)


def corexdc(xs, ys, k=3, base=2, warning=True):
    return tcd(xs, base) - ctcdc(xs, ys, k, base, warning)


# UTILITY FUNCTIONS

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, dualtree=True)[0][:, k]


def query_neighbors_index(tree, x, k):
    return tree.query(x, k=k + 1, dualtree=True, return_distance=False)[:, 1:]


def marginal_distance(points, idx):
    return np.abs(points[idx] - points[:, None, :]).max(axis=(1, 2))


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def avgdigamma2(points, idx):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = marginal_distance(points, idx)
    num_points = count_neighbors(tree, points, dvec) - 1
    return np.mean(digamma(num_points))


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')


@njit
def volume(knn_points):
    knn_points = np.abs(knn_points)
    n, d = knn_points.shape
    res = np.empty(d)
    for i in range(d):
        res[i] = knn_points[:, i].max()
    return np.log(res).sum()

@njit
def pca_volume(knn_points):
    k = knn_points.shape[0]
    # Substract mean of k-nearest neighbor points
    knn_points = knn_points - knn_points[0]
    # Calculate covariance matrix of k-nearest neighbors, get eigen vectors
    covr = knn_points.T @ knn_points / k
    v = np.ascontiguousarray(la.eigh(covr)[1])
    # Calculate PCA-bounding box using eigen vectors
    log_V_rect = volume(knn_points @ v)
    return log_V_rect


# TESTS


def shuffle_test(measure, x, y, z=False, ns=200, ci=0.95, **kwargs):
    """ Shuffle test
        Repeatedly shuffle the x-values and then estimate measure(x, y, [z]).
        Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
        'measure' could me mi, cmi, e.g. Keyword arguments can be passed.
        Mutual information and CMI should have a mean near zero.
    """
    x_clone = np.copy(x)  # A copy that we can shuffle
    outputs = []
    for i in range(ns):
        np.random.shuffle(x_clone)
        if z:
            outputs.append(measure(x_clone, y, z, **kwargs))
        else:
            outputs.append(measure(x_clone, y, **kwargs))
    outputs.sort()
    return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])


if __name__ == "__main__":
    print("MI between two independent continuous random variables X and Y:")
    np.random.seed(0)
    x = np.random.randn(1000, 10)
    y = np.random.randn(1000, 3)
    print(mi(x, y, base=2, alpha=0))
