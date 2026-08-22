"""Load the H3-Optimizations package from the environment or sibling checkout."""

from importlib import import_module
from pathlib import Path
import sys


PACKAGE_NAME = "h3_optimizations"
SIBLING_ROOT = Path(__file__).resolve().parent.parent / "H3-Optimizations"


def _load_package():
    try:
        return import_module(PACKAGE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name != PACKAGE_NAME:
            raise

    package_file = SIBLING_ROOT / PACKAGE_NAME / "__init__.py"
    if not package_file.is_file():
        raise ModuleNotFoundError(
            "ComfyUI-H3-Extended requires H3-Optimizations. Install the "
            "H3-Optimizations custom-node pack beside ComfyUI-H3-Extended."
        )

    sibling = str(SIBLING_ROOT)
    sys.path.insert(0, sibling)
    try:
        return import_module(PACKAGE_NAME)
    finally:
        sys.path.remove(sibling)


PACKAGE = _load_package()


def dependency_module(name):
    return import_module("%s.%s" % (PACKAGE.__name__, name))
