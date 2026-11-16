"""The packages (release/): the icon files, and the files the software reads, in the recipe. The packages themselves
are built and checked by the build machines of GitHub (.github/workflows/build.yml, app/self_check.py).

Skipped if PySide6 is not installed (pip install -r requirements-app.txt).
"""
from pathlib import Path
import struct

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage                    # noqa: E402

from app.i18n import LOCALES                        # noqa: E402
from app.icon import draw                           # noqa: E402
from app.theme import FONTS                         # noqa: E402
from release.icons import ICNS_TYPES, ICO_SIZES, write  # noqa: E402
from simulation.loop import BRAINS                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def same_as_drawn(data: bytes, size: int) -> bool:
    """The PNG `data` is the icon drawn at `size` pixels, to the last pixel."""
    image = QImage.fromData(data, "PNG").convertToFormat(QImage.Format_ARGB32)
    return image.width() == image.height() == size and image == draw(size).convertToFormat(QImage.Format_ARGB32)


def test_icon_files(tmp_path):
    files = write(tmp_path)
    assert {name: path.name for name, path in files.items()} == {"ico": "SAPIR.ico", "icns": "SAPIR.icns"}
    ico = files["ico"].read_bytes()
    assert struct.unpack("<HHH", ico[:6]) == (0, 1, len(ICO_SIZES))
    for i, size in enumerate(ICO_SIZES):
        width, height, _, _, planes, bits, length, offset = struct.unpack("<BBBBHHII", ico[6 + 16 * i: 22 + 16 * i])
        assert (width or 256, height or 256, planes, bits) == (size, size, 1, 32)
        assert same_as_drawn(ico[offset: offset + length], size)
    assert ICO_SIZES[0] == 16 and ICO_SIZES[-1] == 256
    icns = files["icns"].read_bytes()
    assert icns[:4] == b"icns" and struct.unpack(">I", icns[4:8])[0] == len(icns)
    kinds, i = [], 8
    while i < len(icns):
        kind, length = icns[i: i + 4].decode("ascii"), struct.unpack(">I", icns[i + 4: i + 8])[0]
        kinds.append(kind)
        assert same_as_drawn(icns[i + 8: i + length], dict(ICNS_TYPES)[kind])
        i += length
    assert kinds == [kind for kind, _ in ICNS_TYPES] and i == len(icns)


def test_files_read_by_the_software_are_packaged():
    """The folders where the software looks for its files (next to its modules) are in the recipe, at the same place
    relative to the root."""
    recipe = (ROOT / "release" / "sapir.spec").read_text(encoding="utf-8")
    for folder in (FONTS, LOCALES, BRAINS):
        relative = folder.relative_to(ROOT).as_posix()
        assert f'"{relative}")' in recipe and any(folder.iterdir())
