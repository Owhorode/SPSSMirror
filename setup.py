from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="spssmirror",
    version="2.0.0",
    author="SPSSMirror Contributors",
    description="A unified, self-contained statistical analysis library for Python -- an SPSS/R replacement.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Owhorode/SPSSMirror",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    # Core dependencies: required for descriptive stats, regression, the full
    # frequentist suite, psychometrics, effect sizes, power analysis, and
    # diagnostics (Phases 0-7). Kept deliberately lean.
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "statsmodels>=0.14.0",
        "pydantic>=2.0.0",
        "patsy>=0.5.3",
        "rapidfuzz>=2.0.0",
        # Used only for its standalone calculate_kmo/calculate_bartlett_sphericity
        # functions (psychometrics.kmo/bartlett_sphericity) -- NOT its
        # FactorAnalyzer class, which is broken against current scikit-learn.
        # See Phase 5 build notes.
        "factor_analyzer>=0.4.0",
    ],
    extras_require={
        # Mixed models (Phase 8) only needs statsmodels, already core --
        # no separate extra required.
        "bayesian": ["pymc>=5.0.0", "arviz>=0.15.0"],
        "timeseries": ["arch>=6.0.0"],
        "survival": ["lifelines>=0.27.0"],
        "multivariate": ["scikit-learn>=1.2.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov"],
        # Everything at once
        "all": [
            "pymc>=5.0.0", "arviz>=0.15.0", "arch>=6.0.0",
            "lifelines>=0.27.0", "scikit-learn>=1.2.0",
        ],
    },
)
