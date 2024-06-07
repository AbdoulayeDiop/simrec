import numpy as np

from sklearn.utils import check_array, check_random_state

from .distance import check_distance
from .initialization import check_initialization
from .optimization import fit, predict


class KPrototypes:
    """K-Prototypes clustering.

    The k-prototypes algorithm, as described in "Clustering large data sets
    with mixed numeric and categorical values" by Huang (1997), is an extension
    of k-means for mixed data.

    This wrapper loosely follows Scikit-Learn conventions for clustering
    estimators, as it provide the usual ``fit``  and ``predict`` methods.
    However, the signature is different, as it expects numerical and
    categorical data to be provided in separated arrays.

    See Also
    --------
    :meth:`fit`, :meth:`predict`

    Attributes
    ----------
    initialization: callable
        Centroid initialization function.
    numerical_distance: callable
        Distance function used for numerical features.
    categorical_distance: callable
        Distance function used for categorical features.
    gamma: float32
        Categorical distance weight.
    n_clusters: int32
        Number of clusters.
    n_iterations: int32
        Maximum number of iterations.
    verbose: bool
        Verbosity level (0 for no output).
    true_gamma: float32
        Categorical distance weight inferred from data, if gamma was not
        specified.
    numerical_centroids: float32, n_clusters x n_numerical_features
        Numerical centroid array.
    categorical_centroids: int32, n_clusters x n_categorical_features
        Categorical centroid array.
    cost: float32
        Loss after last training iteration.

    """

    def __init__(
        self,
        n_clusters=8,
        initialization=None,
        numerical_distance="euclidean",
        categorical_distance="matching",
        gamma=None,
        n_iterations=100,
        random_state=None,
        verbose=0,
    ):

        # Resolve string-based properties
        self.initialization = check_initialization(initialization)
        self.numerical_distance = check_distance(numerical_distance)
        self.categorical_distance = check_distance(categorical_distance)

        # Gamma and random state will be resolved when fitted
        self.gamma = gamma
        self.random_state = random_state

        # Store other arguments, ensuring type
        self.n_clusters = int(n_clusters)
        self.n_iterations = int(n_iterations)
        self.verbose = bool(verbose)

        # Parameters are not yet fitted
        self.true_gamma = None
        self.numerical_centroids = None
        self.categorical_centroids = None
        self.cost = None

    def fit(self, numerical_values, categorical_values):
        """Fit centroids.

        Parameters
        ----------
        numerical_values: float32, n_samples x n_numerical_features
            Numerical feature array.
        categorical_values: int32, n_samples x n_categorical_features
            Categorical feature array.

        Returns
        -------
        self: object

        """

        # Regular fit, discarding cluster assignment
        self.fit_predict(numerical_values, categorical_values)
        return self

    def fit_predict(self, numerical_values, categorical_values):
        """Fit centroids and assign points to closest clusters.

        Parameters
        ----------
        numerical_values: float32, n_samples x n_numerical_features
            Numerical feature array.
        categorical_values: int32, n_samples x n_categorical_features
            Categorical feature array.

        Returns
        -------
        clustership: int32, n_samples
            Closest clusers.

        """

        # Check input
        # TODO maybe ensure_min_features=0?
        numerical_values = check_array(
            numerical_values, dtype=[np.float32, np.float64],
        )
        categorical_values = check_array(
            categorical_values, dtype=[np.int32, np.int64],
        )

        # Estimate gamma, if not specified
        if self.gamma is None:
            gamma = 0.5 * numerical_values.std()
        else:
            gamma = float(self.gamma)

        # Resolve random state
        random_state = check_random_state(self.random_state)

        # Initialize clusters
        numerical_centroids, categorical_centroids = self.initialization(
            numerical_values,
            categorical_values,
            self.n_clusters,
            self.numerical_distance,
            self.categorical_distance,
            gamma,
            random_state,
            self.verbose,
        )

        # Train clusters
        clustership, cost = fit(
            numerical_values,
            categorical_values,
            numerical_centroids,
            categorical_centroids,
            self.numerical_distance,
            self.categorical_distance,
            gamma,
            self.n_iterations,
            random_state,
            self.verbose,
        )

        # Save result
        self.true_gamma = gamma
        self.numerical_centroids = numerical_centroids
        self.categorical_centroids = categorical_centroids
        self.cost = cost

        return clustership

    def predict(self, numerical_values, categorical_values):
        """Assign points to closest clusters.

        Parameters
        ----------
        numerical_values: float32, n_samples x n_numerical_features
            Numerical feature array.
        categorical_values: int32, n_samples x n_categorical_features
            Categorical feature array.

        Returns
        -------
        clustership: int32, n_samples
            Closest clusers.

        """

        return predict(
            numerical_values,
            categorical_values,
            self.numerical_centroids,
            self.categorical_centroids,
            self.numerical_distance,
            self.categorical_distance,
            self.true_gamma,
        )
