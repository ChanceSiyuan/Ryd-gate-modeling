"""Session fixtures for writable third-party caches."""

import os
import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def writable_arc_database(tmp_path_factory):
    """Run ARC against a temporary copy instead of a user's SQLite cache."""
    matplotlib_dir = tmp_path_factory.mktemp("matplotlib")
    previous_matplotlib_dir = os.environ.get("MPLCONFIGDIR")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    from arc import AlkaliAtom, PairStateInteractions

    source = AlkaliAtom.dataFolder
    target = tmp_path_factory.mktemp("arc-data")
    shutil.copytree(source, target, dirs_exist_ok=True)

    classes = (AlkaliAtom, PairStateInteractions)
    previous_data_folders = {cls: cls.dataFolder for cls in classes}
    for cls in classes:
        cls.dataFolder = str(target)

    try:
        yield
    finally:
        for cls, data_folder in previous_data_folders.items():
            cls.dataFolder = data_folder
        if previous_matplotlib_dir is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_matplotlib_dir
