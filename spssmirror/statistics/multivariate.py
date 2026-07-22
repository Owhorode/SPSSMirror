from typing import List, Optional
import numpy as np
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column, validate_min_sample_size
from spssmirror.models._results import (
    PCAComponent, PCAResult, ClusterProfile, ClusteringResult,
    DiscriminantFunctionCoefficient, DiscriminantResult, CanonicalCorrelationResult,
)


class MultivariateEngine:
    """
    PCA, clustering (k-means and hierarchical), discriminant analysis
    (LDA/QDA), and canonical correlation. scikit-learn/scipy are used
    internally only — every method returns a SPSSMirror result model.

    Note: exploratory factor analysis already lives in the psychometrics
    engine (Phase 5), implemented via manual eigen-decomposition + varimax
    rather than the third-party `factor_analyzer` package (which is broken
    against current scikit-learn — see Phase 5 build notes). It is not
    duplicated here.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _clean_matrix(self, columns: List[str]) -> np.ndarray:
        for c in columns:
            validate_numeric_column(self._engine.get_column(c), c)
        df_clean = self._engine.to_dataframe()[columns].dropna()
        validate_min_sample_size(len(df_clean), len(columns) + 2, "Multivariate analysis")
        return df_clean.to_numpy(dtype=float), df_clean

    def pca(self, columns: List[str], n_components: Optional[int] = None,
            standardize: bool = True) -> PCAResult:
        from sklearn.decomposition import PCA as SKPCA
        from sklearn.preprocessing import StandardScaler

        if len(columns) < 2:
            raise ValueError("PCA requires at least 2 variables.")

        X, df_clean = self._clean_matrix(columns)
        n = X.shape[0]
        k = n_components or len(columns)
        k = min(k, len(columns), n)

        if standardize:
            X = StandardScaler().fit_transform(X)

        try:
            pca = SKPCA(n_components=k)
            scores = pca.fit_transform(X)
        except Exception as e:
            raise ValueError(f"PCA failed: {e}") from e

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        components = []
        for i in range(k):
            loadings = {columns[j]: float(pca.components_[i, j]) for j in range(len(columns))}
            components.append(PCAComponent(
                component=f"PC{i + 1}", explained_variance=float(pca.explained_variance_[i]),
                explained_variance_ratio=float(pca.explained_variance_ratio_[i]),
                cumulative_variance_ratio=float(cumulative[i]), loadings=loadings,
            ))

        return PCAResult(variables=columns, components=components, scores=scores.tolist(), n=n)

    def kmeans_clustering(self, columns: List[str], n_clusters: int = 3,
                           standardize: bool = True, seed: int = 0) -> ClusteringResult:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        X, df_clean = self._clean_matrix(columns)
        n = X.shape[0]
        if n_clusters >= n:
            raise ValueError(f"n_clusters ({n_clusters}) must be less than the number of observations ({n}).")

        X_scaled = StandardScaler().fit_transform(X) if standardize else X

        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
            labels = km.fit_predict(X_scaled)
        except Exception as e:
            raise ValueError(f"K-means clustering failed: {e}") from e

        silhouette = float(silhouette_score(X_scaled, labels)) if n_clusters < n else None

        profiles = []
        for c in range(n_clusters):
            mask = labels == c
            centroid = {col: float(df_clean.iloc[:, j][mask].mean()) for j, col in enumerate(columns)}
            profiles.append(ClusterProfile(cluster=str(c), n=int(mask.sum()), centroid=centroid))

        return ClusteringResult(
            method="K-Means", variables=columns, labels=labels.tolist(), profiles=profiles,
            n_clusters=n_clusters, inertia=float(km.inertia_), silhouette=silhouette, n=n,
        )

    def hierarchical_clustering(self, columns: List[str], n_clusters: int = 3,
                                 linkage_method: str = "ward", standardize: bool = True) -> ClusteringResult:
        from scipy.cluster.hierarchy import linkage, fcluster
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        valid_methods = ("ward", "complete", "average", "single")
        if linkage_method not in valid_methods:
            raise ValueError(f"linkage_method must be one of {valid_methods}. Got '{linkage_method}'.")
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        X, df_clean = self._clean_matrix(columns)
        n = X.shape[0]
        if n_clusters >= n:
            raise ValueError(f"n_clusters ({n_clusters}) must be less than the number of observations ({n}).")

        X_scaled = StandardScaler().fit_transform(X) if standardize else X

        try:
            Z = linkage(X_scaled, method=linkage_method)
            labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # zero-index
        except Exception as e:
            raise ValueError(f"Hierarchical clustering failed: {e}") from e

        silhouette = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else None

        profiles = []
        for c in sorted(set(labels)):
            mask = labels == c
            centroid = {col: float(df_clean.iloc[:, j][mask].mean()) for j, col in enumerate(columns)}
            profiles.append(ClusterProfile(cluster=str(c), n=int(mask.sum()), centroid=centroid))

        return ClusteringResult(
            method=f"Hierarchical ({linkage_method})", variables=columns, labels=labels.tolist(),
            profiles=profiles, n_clusters=len(set(labels)), silhouette=silhouette,
            linkage_matrix=Z.tolist(), n=n,
        )

    def _discriminant(self, predictors: List[str], group_col: str, method: str) -> DiscriminantResult:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix

        cols = predictors + [group_col]
        for c in predictors:
            validate_numeric_column(self._engine.get_column(c), c)
        df_clean = self._engine.to_dataframe()[cols].dropna()
        n = len(df_clean)
        validate_min_sample_size(n, len(predictors) + 5, f"{method} discriminant analysis")

        groups = sorted(df_clean[group_col].astype(str).unique().tolist())
        if len(groups) < 2:
            raise ValueError(f"'{group_col}' must have at least 2 groups for discriminant analysis.")

        X = df_clean[predictors].to_numpy(dtype=float)
        y = df_clean[group_col].astype(str).to_numpy()

        model = LinearDiscriminantAnalysis() if method == "linear" else QuadraticDiscriminantAnalysis()
        try:
            model.fit(X, y)
            predictions = model.predict(X)
        except Exception as e:
            raise ValueError(f"{method.capitalize()} discriminant analysis failed to fit: {e}") from e

        accuracy = float((predictions == y).mean())
        cm = sk_confusion_matrix(y, predictions, labels=groups)

        coefficients = []
        if method == "linear" and hasattr(model, "coef_"):
            coef_arr = model.coef_
            for j, var in enumerate(predictors):
                if coef_arr.shape[0] == 1:
                    coefficients.append(DiscriminantFunctionCoefficient(
                        variable=var, coefficients={"LD1": float(coef_arr[0, j])},
                    ))
                else:
                    coefficients.append(DiscriminantFunctionCoefficient(
                        variable=var,
                        coefficients={f"LD{i + 1}": float(coef_arr[i, j]) for i in range(coef_arr.shape[0])},
                    ))

        explained_var_ratio = []
        if method == "linear" and hasattr(model, "explained_variance_ratio_"):
            explained_var_ratio = [float(v) for v in model.explained_variance_ratio_]

        return DiscriminantResult(
            method=f"{method.capitalize()} Discriminant Analysis", group_col=group_col,
            predictors=predictors, groups=groups, accuracy=accuracy, coefficients=coefficients,
            explained_variance_ratio=explained_var_ratio, confusion_matrix=cm.tolist(), n=n,
        )

    def linear_discriminant(self, predictors: List[str], group_col: str) -> DiscriminantResult:
        return self._discriminant(predictors, group_col, method="linear")

    def quadratic_discriminant(self, predictors: List[str], group_col: str) -> DiscriminantResult:
        return self._discriminant(predictors, group_col, method="quadratic")

    def canonical_correlation(self, set1: List[str], set2: List[str],
                               n_components: Optional[int] = None) -> CanonicalCorrelationResult:
        from sklearn.cross_decomposition import CCA

        if len(set1) < 1 or len(set2) < 1:
            raise ValueError("Both variable sets must be non-empty.")
        overlap = set(set1) & set(set2)
        if overlap:
            raise ValueError(f"Variable sets must not overlap. Shared columns: {overlap}")

        all_cols = set1 + set2
        for c in all_cols:
            validate_numeric_column(self._engine.get_column(c), c)
        df_clean = self._engine.to_dataframe()[all_cols].dropna()
        n = len(df_clean)
        k = n_components or min(len(set1), len(set2))
        k = min(k, len(set1), len(set2))
        validate_min_sample_size(n, max(len(set1), len(set2)) + 5, "Canonical correlation")

        X = df_clean[set1].to_numpy(dtype=float)
        Y = df_clean[set2].to_numpy(dtype=float)

        try:
            cca = CCA(n_components=k)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
        except Exception as e:
            raise ValueError(f"Canonical correlation analysis failed: {e}") from e

        correlations = []
        for i in range(k):
            x_col, y_col = X_c[:, i], Y_c[:, i]
            if np.std(x_col) > 0 and np.std(y_col) > 0:
                r = float(np.corrcoef(x_col, y_col)[0, 1])
            else:
                r = 0.0
            correlations.append(r)

        return CanonicalCorrelationResult(
            set1_variables=set1, set2_variables=set2, canonical_correlations=correlations,
            n_pairs=k, n=n,
        )
