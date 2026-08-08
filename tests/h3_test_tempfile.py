"""Test-only temporary directories with inherited Windows permissions."""

from pathlib import Path
import shutil
import uuid


_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / ".agent" / "tmp" / "tests"


def gettempdir():
    _DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
    return str(_DEFAULT_ROOT)


def mkdtemp(suffix="", prefix="tmp", dir=None):
    root = _DEFAULT_ROOT if dir is None else Path(dir)
    if dir is None:
        gettempdir()

    suffix = "" if suffix is None else suffix
    prefix = "tmp" if prefix is None else prefix
    for _ in range(100):
        path = root / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return str(path)
    raise FileExistsError("could not create a unique temporary directory")


class TemporaryDirectory:
    def __init__(self, suffix="", prefix="tmp", dir=None):
        self.name = mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self._closed = False

    def cleanup(self):
        if self._closed:
            return
        try:
            shutil.rmtree(self.name)
        except FileNotFoundError:
            pass
        finally:
            self._closed = True

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def __del__(self):
        self.cleanup()
