from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]


class AdultPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, add_features=True, lower_quantile=0.01, upper_quantile=0.99):
        self.add_features = add_features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def _inputs(self, frame):
        if list(frame.columns) != FEATURES:
            raise ValueError("Expected the 14 Adult features in their original order.")
        numeric = frame[NUMERIC].replace("?", np.nan).astype(np.float64)
        categorical = frame[CATEGORICAL].astype(object)
        categorical = categorical.where(categorical.notna() & categorical.ne("?"), np.nan)
        return numeric, categorical

    def _numeric_features(self, numeric):
        clipped = np.clip(numeric, self.lower_bounds_, self.upper_bounds_)
        if not self.add_features:
            return clipped
        denominator = clipped[:, 0] - 16
        if np.any(denominator <= 0):
            raise ValueError("hours_per_age requires age greater than 16.")
        capital_net = clipped[:, 3] - clipped[:, 4]
        hours_per_age = clipped[:, 5] / denominator
        return np.column_stack((clipped, capital_net, hours_per_age))

    def fit(self, frame, y=None):
        numeric, categorical = self._inputs(frame)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        self.numeric_imputer_ = SimpleImputer(strategy="median").fit(numeric)
        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent").fit(categorical)
        imputed = self.numeric_imputer_.transform(numeric)
        self.lower_bounds_, self.upper_bounds_ = np.quantile(
            imputed, [self.lower_quantile, self.upper_quantile], axis=0
        )
        self.scaler_ = StandardScaler().fit(self._numeric_features(imputed))
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        self.encoder_.fit(self.categorical_imputer_.transform(categorical))
        return self

    def transform(self, frame):
        check_is_fitted(self, "encoder_")
        numeric, categorical = self._inputs(frame)
        numeric = self._numeric_features(self.numeric_imputer_.transform(numeric))
        categorical = self.encoder_.transform(self.categorical_imputer_.transform(categorical))
        return np.column_stack((self.scaler_.transform(numeric), categorical))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoder_")
        numeric = NUMERIC + (["capital_net", "hours_per_age"] if self.add_features else [])
        return np.asarray(numeric + list(self.encoder_.get_feature_names_out(CATEGORICAL)), dtype=str)
