from collections import Counter

import numpy as np


UNKNOWN = object()


class SingleCategoricalTransformer:
    """Transform a single column."""

    def __init__(self, *, min_count=0, allow_unknown=True, nan_as_unknown=True):

        # Keep parameters
        if min_count > 0 and not allow_unknown:
            raise ValueError("Cannot use min_count when unknown values are forbidden")
        self.min_count = min_count
        self.allow_unknown = allow_unknown
        self.nan_as_unknown = nan_as_unknown

        # Parameters are not yet fitted
        self._table = None
        self._mapping = None

    def fit(self, values):

        # Count values, unifying NaN-likes
        counter = Counter()
        for value in values:
            if value != value:
                value = np.nan
            counter[value] += 1

        # Discard NaN, if considered as unknown value
        if self.nan_as_unknown and np.nan in counter:
            if not self.allow_unknown:
                raise ValueError(
                    "Cannot have NaN-like values as unknown when unknown "
                    "values are forbidden"
                )
            del counter[np.nan]

        # Create mapping table
        table = [v for v, c in counter.items() if c >= self.min_count]
        if self.allow_unknown:
            mapping = {v: i + 1 for i, v in enumerate(table)}
            table = [UNKNOWN, *table]
        else:
            mapping = {v: i for i, v in enumerate(table)}

        # Define mapping function
        if self.allow_unknown:

            def _map(value):
                if value != value:
                    value = np.nan
                return mapping.get(value, 0)

        else:

            def _map(value):
                if value != value:
                    value = np.nan
                return mapping[value]

        # Wrap as NumPy objects
        self._table = np.array(table, dtype=object)
        self._mapping = np.vectorize(_map, otypes=[np.int32])

        return self

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def transform(self, values):
        return self._mapping(values)

    def inverse_transform(self, indices):
        return self._table[indices]


class CategoricalTransformer:
    """Encode categorical values as integers.

    Each column has its own vocabulary. Values are mapped from 0 to N - 1,
    where N is the size of the vocabulary.

    Parameters
    ----------
    min_count: int, optional
        Ignore values that appears less than a given number of times. Unknown
        values must be enabled as well.
    allow_unknown: bool, optional
        Add an additional value for unexpected or unknown values.
    nan_as_unknown: bool, optional
        Treat NaN as unknown, instead of allocating a dedicated index.

    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._transformers = []

    def fit(self, values):
        """Build index."""

        _, n_columns = values.shape
        transformers = []
        for i in range(n_columns):
            transformer = SingleCategoricalTransformer(**self._kwargs)
            transformer.fit(values[:, i])
            transformers.append(transformer)
        self._transformers = transformers
        return self

    def fit_transform(self, values):
        """Build index and transform values."""

        self.fit(values)
        return self.transform(values)

    def transform(self, values):
        """Transform values."""

        indices = [t.transform(values[:, i]) for i, t in enumerate(self._transformers)]
        indices = np.stack(indices, axis=1)
        return indices

    def inverse_transform(self, indices):
        """Convert indices back to values."""

        values = [
            t.inverse_transform(indices[:, i]) for i, t in enumerate(self._transformers)
        ]
        values = np.stack(values, axis=1)
        return values
