from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
__path__ = [str(_PACKAGE_ROOT)] + list(__path__)

from .core import SPSSMirror
from .models import (
    DataQuality,
    StatTestResult,
    CorrelationResult,
    RegressionResult,
    PsychometricResult,
    FrequencyTableResult,
    DescriptiveResult,
    CrossTabResult,
)
from ._version import __version__

__all__ = [
    "SPSSMirror",
    "DataQuality",
    "StatTestResult",
    "CorrelationResult",
    "RegressionResult",
    "PsychometricResult",
    "FrequencyTableResult",
    "DescriptiveResult",
    "CrossTabResult",
    "__version__",
]
