import numpy as np

from sklearn.utils import check_array


def check_initialization(initialization):
    """Resolve initialization function.

    If ``distance`` is a string, only ``"random"`` and ``"frequency"`` are
    accepted. If it is a callable, then is is return as-is. If it is ``None``,
    it defaults to ``"random"``.

    Directly specifying centroids as a tuple of arrays is also accepted.

    Returns
    -------
    function: callable
        Centroid initialization function.

    See Also
    --------
    :meth:`random_initialization`, :meth:`frequency_initialization`

    """

    if initialization is None:
        return random_initialization
    if callable(initialization):
        return initialization
    if initialization == "random":
        return random_initialization
    if initialization == "frequency":
        return frequency_initialization
    if isinstance(initialization, (tuple, list)):
        assert len(initialization) == 2
        return _explicit_initialization_factory(*initialization)
    raise KeyError(initialization)


def _explicit_initialization_factory(numerical_centroids, categorical_centroids):
    """Create dummy initialization method, returning precomputed centroids."""

    # Check types, and copy arrays
    numerical_centroids = check_array(
        numerical_centroids, dtype=[np.float32, np.float64], copy=True,
    )
    categorical_centroids = check_array(
        categorical_centroids, dtype=[np.int32, np.int64], copy=True,
    )

    # Check shape
    assert len(numerical_centroids.shape) == 2
    assert len(categorical_centroids.shape) == 2
    assert numerical_centroids.shape[0] == categorical_centroids.shape[0]

    def initialization(
        numerical_values,
        categorical_values,
        n_clusters,
        numerical_distance,
        categorical_distance,
        gamma,
        random_state,
        verbose,
    ):

        # Check number of cluster
        assert numerical_centroids.shape[0] == n_clusters
        assert categorical_centroids.shape[0] == n_clusters

        # Check number of features
        assert numerical_centroids.shape[1] == numerical_values.shape[1]
        assert categorical_centroids.shape[1] == categorical_values.shape[1]

        return numerical_centroids, categorical_centroids

    return initialization


def random_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_distance,
    categorical_distance,
    gamma,
    random_state,
    verbose,
):
    """Random initialization.

    Choose random points as cluster centroids.

    Used in "Clustering large data sets with mixed numeric and categorical
    values" by Huang (1997), the original k-prototypes definition.

    Returns
    -------
    numerical_centroids: float32, n_clusters x n_numerical_features
        Numerical centroid array.
    categorical_centroids: int32, n_clusters x n_categorical_features
        Categorical centroid array.

    """

    n_points, _ = numerical_values.shape
    assert n_points == categorical_values.shape[0]
    assert n_points >= n_clusters

    # TODO need to discard duplicates?
    indices = random_state.permutation(n_points)
    selected_indices = indices[:n_clusters]
    numerical_centroids = numerical_values[selected_indices]
    categorical_centroids = categorical_values[selected_indices]
    return numerical_centroids, categorical_centroids


# TODO "Extensions to the k-modes algorithm for clustering large data sets with categorical values" by Huang (1998)?


# TODO histogram-based kde


def _numerical_density_sklearn(values):
    """Estimate probability density function using gaussian kernel.

    Requires ``scikit-learn``.

    """

    from sklearn.neighbors import KernelDensity

    v = values[:, None]
    kde = KernelDensity()
    # TODO randomly subsample if too large?
    kde.fit(v)
    log_densities = kde.score_samples(v)
    densities = np.exp(log_densities)
    return densities


def _numerical_density_fastkde(values):
    """Estimate probability density function using a fast approximation.

    Requires ``fastKDE``, as proposed by O'Brien et al. in "A fast and
    objective multidimensional kernel density estimation method: fastKDE".

    """

    from fastkde import fastKDE

    pdf, axe = fastKDE.pdf(values)
    densities = np.interp(values, axe, pdf, left=0, right=0)
    return densities


def _numerical_density(values, method="fast"):
    """Estimate density of a continous random variable."""

    n_points, n_features = values.shape
    densities = np.zeros((n_points, n_features), dtype=np.float32)
    for j in range(n_features):
        densities[:, j] = _numerical_density_fastkde(values[:, j])
    return densities


def _categorical_density(values):
    """Estimate density of a discrete random variable."""

    n_points, n_features = values.shape
    densities = np.zeros((n_points, n_features), dtype=np.int32)
    for j in range(n_features):
        frequencies = np.bincount(values[:, j])
        densities[:, j] = frequencies[values[:, j]]
    densities = densities.astype(np.float32)
    densities /= n_points
    return densities


def frequency_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_distance,
    categorical_distance,
    gamma,
    random_state,
    verbose,
):
    """Frequency-based initialization.

    Choose centroids from points, based on probability distributions of each
    feature. The first centroid is selected at highest density point. Then, the
    remaining centroids are selected to be both far from current centroids and
    at dense locations.

    This is an extension for mixed values of "A new initialization method for
    categorical data clustering" by Cao et al. (2009).

    Returns
    -------
    numerical_centroids: float32, n_clusters x n_numerical_features
        Numerical centroid array.
    categorical_centroids: int32, n_clusters x n_categorical_features
        Categorical centroid array.

    """

    n_points, n_numerical_features = numerical_values.shape
    _, n_categorical_features = categorical_values.shape
    assert n_points == categorical_values.shape[0]
    assert n_points >= n_clusters

    # Allocate centroid arrays
    numerical_centroids = np.empty(
        (n_clusters, n_numerical_features), dtype=numerical_values.dtype,
    )
    categorical_centroids = np.empty(
        (n_clusters, n_categorical_features), dtype=categorical_values.dtype,
    )

    # Estimate probability of each sample and each feature
    densities = np.concatenate(
        [
            _numerical_density(numerical_values),
            _categorical_density(categorical_values),
        ],
        axis=1,
    )

    # Mean density is used as weight
    weights = densities.mean(axis=1)

    # First cluster is most likely point
    index = np.argmax(weights)
    numerical_centroids[0] = numerical_values[index]
    categorical_centroids[0] = categorical_values[index]

    # Then, choose the most dissimilar point at each step, with respect to current cluster set
    for k in range(1, n_clusters):

        # Compute distance w.r.t. already initialized centroids
        numerical_costs = numerical_distance(
            numerical_values[:, None], numerical_centroids[None, :k]
        )
        categorical_costs = categorical_distance(
            categorical_values[:, None], categorical_centroids[None, :k]
        )
        costs = numerical_costs + gamma * categorical_costs

        # Maximize minimum distance (i.e. ensure largest margin)
        weighted_costs = costs * weights[:, None]
        min_weighted_costs = weighted_costs.min(axis=1)
        index = np.argmax(min_weighted_costs)
        numerical_centroids[k] = numerical_values[index]
        categorical_centroids[k] = categorical_values[index]

    return numerical_centroids, categorical_centroids
