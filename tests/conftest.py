import numpy as np
import pandas as pd
import pytest
from spssmirror import SPSSMirror


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def two_group_data(rng):
    """Two clearly different groups: A ~ N(50,10), B ~ N(58,10)."""
    a = rng.normal(50, 10, 60)
    b = rng.normal(58, 10, 60)
    return SPSSMirror().load_dict({
        "score": np.concatenate([a, b]).tolist(),
        "grp": ["A"] * 60 + ["B"] * 60,
    })


@pytest.fixture
def three_group_data(rng):
    """Three groups where A != B, A == C (for post-hoc discrimination)."""
    a = rng.normal(50, 5, 40)
    b = rng.normal(58, 5, 40)
    c = rng.normal(50.5, 5, 40)
    return SPSSMirror().load_dict({
        "score": np.concatenate([a, b, c]).tolist(),
        "grp": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
    })


@pytest.fixture
def linear_regression_data(rng):
    """y = 2 + 3*x1 - 1.5*x2 + noise."""
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 + 3 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
    return SPSSMirror().load_dict({"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()})


@pytest.fixture
def correlated_pair_data(rng):
    """x, y correlated at approximately r=0.8."""
    n = 200
    x = rng.normal(0, 1, n)
    y = 0.7 * x + rng.normal(0, 0.5, n)
    return SPSSMirror().load_dict({"x": x.tolist(), "y": y.tolist()})


@pytest.fixture
def reliable_scale_data(rng):
    """6 items all loading on one shared trait -> high reliability."""
    n = 300
    trait = rng.normal(0, 1, n)
    items = {f"q{i}": (0.75 * trait + rng.normal(0, 0.5, n)).tolist() for i in range(6)}
    return SPSSMirror().load_dict(items), items


@pytest.fixture
def two_factor_data(rng):
    """3 items load on factor1, 3 items load on factor2 -> simple structure."""
    n = 400
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    data = {
        "a1": (0.8 * f1 + rng.normal(0, 0.5, n)).tolist(),
        "a2": (0.8 * f1 + rng.normal(0, 0.5, n)).tolist(),
        "a3": (0.8 * f1 + rng.normal(0, 0.5, n)).tolist(),
        "b1": (0.8 * f2 + rng.normal(0, 0.5, n)).tolist(),
        "b2": (0.8 * f2 + rng.normal(0, 0.5, n)).tolist(),
        "b3": (0.8 * f2 + rng.normal(0, 0.5, n)).tolist(),
    }
    return SPSSMirror().load_dict(data), list(data.keys())


@pytest.fixture
def ar1_series(rng):
    """AR(1) process with true coefficient 0.7."""
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + rng.normal(0, 1)
    return SPSSMirror().load_dict({"y": y.tolist()})


@pytest.fixture
def survival_two_group_data(rng):
    """Group A engineered to outlive Group B."""
    n = 100
    dur_a = rng.exponential(scale=20, size=n)
    dur_b = rng.exponential(scale=5, size=n)
    return SPSSMirror().load_dict({
        "duration": np.concatenate([dur_a, dur_b]).tolist(),
        "event": np.ones(2 * n).tolist(),
        "grp": ["A"] * n + ["B"] * n,
    })


@pytest.fixture
def separable_clusters_data(rng):
    """Three well-separated 2D clusters."""
    n_per = 60
    a = rng.normal([0, 0], 0.5, (n_per, 2))
    b = rng.normal([10, 10], 0.5, (n_per, 2))
    c = rng.normal([0, 10], 0.5, (n_per, 2))
    X = np.vstack([a, b, c])
    return SPSSMirror().load_dict({"x": X[:, 0].tolist(), "y": X[:, 1].tolist()})
