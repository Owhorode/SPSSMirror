import importlib
import sys
from pathlib import Path


def test_forest_analysis_module_imports_and_resolves_data_file():
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    module = importlib.import_module("forest_analysis")

    assert module.DATA_PATH.exists()
    assert module.DATA_PATH.name == "forest_conservation_response_data.csv"
