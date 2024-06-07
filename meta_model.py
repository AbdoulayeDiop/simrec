from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

class KNN(KNeighborsRegressor):
    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform", **params) -> None:
        super().__init__(n_neighbors=n_neighbors, metric=metric, weights=weights, **params)

def create_pipeline(scaler=None, selected_features=None, meta_model=KNN()):
    steps = []
    if selected_features is not None:
        feature_selector = ColumnTransformer([("features_selection", "passthrough", selected_features)])
        steps.append(("features_selector", feature_selector))
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("meta_model", meta_model))
    return Pipeline(steps)

if __name__ == "__main__":
    import numpy as np
    X, Y = np.random.rand(100, 8), np.random.rand(100, 2)
    predictor = KNeighborsRegressor(n_neighbors=5)
    scaler=StandardScaler()
    model = create_pipeline(scaler=scaler)
    model.fit(X, Y)
    print(model.get_params())
