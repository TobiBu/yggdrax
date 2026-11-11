"""The explicit setuptools package list must name every subpackage.

``pyproject.toml`` lists packages by hand, so a new subpackage that is not added
there is silently left out of built wheels while every test still passes from
the source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

_ROOT = Path(__file__).resolve().parents[2]


def test_every_subpackage_is_in_the_setuptools_package_list():
    with open(_ROOT / "pyproject.toml", "rb") as f:
        listed = set(tomllib.load(f)["tool"]["setuptools"]["packages"])
    on_disk = {
        ".".join(p.parent.relative_to(_ROOT).parts)
        for p in (_ROOT / "yggdrax").rglob("__init__.py")
    }
    assert on_disk - listed == set(), "add these to [tool.setuptools] packages"
    assert listed - on_disk == set(), "these listed packages do not exist"
