import numpy as np


def _iterations(n_iterations):
    i = 0
    while n_iterations < 1 or i < n_iterations:
        yield i
        i += 1


def fit(
    numerical_values,
    categorical_values,
    numerical_centroids,
    categorical_centroids,
    numerical_distance,
    categorical_distance,
    gamma,
    n_iterations,
    random_state,
    verbose,
):
    """Fit centroids.

    This implementation follows the standard k-means algorithm, also referred
    to as Lloyd's algorithm. The optimization proceeds by alternating between
    two steps:
    
     1. assignment step, where each sample is assigned to the closest
        centroid;

     2. update step, where centroids are recomputed based on the assignment.

    This approach differs from the original paper, where centroids are updated
    after each individual assignment.

    Parameters
    ----------
    numerical_values: float32, n_samples x n_numerical_features
        Numerical feature array.
    categorical_values: int32, n_samples x n_categorical_features
        Categorical feature array.
    numerical_centroids: float32, n_clusters x n_numerical_features
        Numerical centroid array.
    categorical_centroids: int32, n_clusters x n_categorical_features
        Categorical centroid array.
    numerical_distance: callable
        Distance function used for numerical features.
    categorical_distance: callable
        Distance function used for categorical features.
    gamma: float32
        Categorical distance weight.
    n_iterations: int32
        Maximum number of iterations.
    random_state: numpy.random.RandomState
        Random state used to initialize centroids.
    verbose: int32
        Verbosity level (0 for no output).

    Returns
    -------
    clustership: int32, n_samples
        Closest clusers.
    cost: float32
        Loss after last iteration.

    """

    n_points, n_categorical_features = categorical_values.shape
    n_clusters, _ = numerical_centroids.shape

    # TODO maybe allow keyboard interrupt?

    clustership = None
    cost = np.inf
    for iteration in _iterations(n_iterations):
        old_clustership = clustership

        # Assign points to closest clusters
        clustership, cost = predict(
            numerical_values,
            categorical_values,
            numerical_centroids,
            categorical_centroids,
            numerical_distance,
            categorical_distance,
            gamma,
            return_cost=True,
        )

        # Check for convergence
        if old_clustership is not None:
            moves = (old_clustership != clustership).sum()
            if verbose > 0:
                print(f"#{iteration}: cost={cost}, moves={moves}")
            if moves == 0:  # TODO abort if cost > old_cost?
                break

        # Count points in each cluster
        masks = clustership[None, :] == np.arange(n_clusters)[:, None]
        counts = masks.sum(axis=1)

        # Update clusters
        for k in range(n_clusters):
            mask = masks[k]
            count = counts[k]

            # If cluster is empty, reinitialize with a random point from largest cluster
            if count == 0:
                largest_cluster = counts.argmax()
                mask = clustership == largest_cluster
                available_points = np.arange(n_points)[mask]
                point = random_state.choice(available_points)
                numerical_centroids[k] = numerical_values[point]
                categorical_centroids[k] = categorical_values[point]

            else:

                # Numerical centroid attributes are set to mean
                masked_numerical_values = numerical_values[mask]
                numerical_centroids[k] = masked_numerical_values.sum(axis=0) / count

                # Categorical centroid attributes are set to most frequent value
                masked_categorical_values = categorical_values[mask]
                for j in range(n_categorical_features):
                    frequency = np.bincount(masked_categorical_values[:, j])
                    categorical_centroids[k, j] = frequency.argmax()

    # Report non-convergence
    else:
        if verbose > 0:
            print(f"Optimization did not converge after {n_iterations} iterations")

    return clustership, cost


def predict(
    numerical_values,
    categorical_values,
    numerical_centroids,
    categorical_centroids,
    numerical_distance,
    categorical_distance,
    gamma,
    return_cost=False,
):
    """Assign points to closest clusters.

    Parameters
    ----------
    numerical_values: float32, n_samples x n_numerical_features
        Numerical feature array.
    categorical_values: int32, n_samples x n_categorical_features
        Categorical feature array.
    numerical_centroids: float32, n_clusters x n_numerical_features
        Numerical centroid array.
    categorical_centroids: int32, n_clusters x n_categorical_features
        Categorical centroid array.
    numerical_distance: callable
        Distance function used for numerical features.
    categorical_distance: callable
        Distance function used for categorical features.
    gamma: float32
        Categorical distance weight.
    return_cost: bool, optional
        Whether to return cost.

    Returns
    -------
    clustership: int32, n_samples
        Closest clusers.
    cost: float32
        Loss after last iteration, if ``return_cost`` is true.

    """

    n_points, _ = numerical_values.shape

    # Compute weighted distances
    numerical_costs = numerical_distance(
        numerical_values, numerical_centroids
    )
    categorical_costs = categorical_distance(
        categorical_values, categorical_centroids
    )
    costs = numerical_costs + gamma * categorical_costs

    # Assign to closest clusters
    clustership = np.argmin(costs, axis=1)

    # Compute cost
    if return_cost:
        cost = costs[np.arange(n_points), clustership].sum()
        return clustership, cost
    return clustership
