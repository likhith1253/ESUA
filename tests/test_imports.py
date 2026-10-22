"""Every module and phase script must at least compile; the library parts must import without a camera."""
import glob
import os
import py_compile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)))
def test_compiles(path, tmp_path):
    if os.path.relpath(path, ROOT).startswith("tests"):
        pytest.skip("test file")
    py_compile.compile(path, cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_core_imports_without_heavy_deps():
    import importlib
    import sys
    sys.path.insert(0, ROOT)
    for m in ("categories", "spatial", "risk_rules", "explanations", "aggregation"):
        importlib.import_module(f"esua_core.{m}")
