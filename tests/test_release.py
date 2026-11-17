"""The packages and the publication (release/): the icon files, the files the software reads in the recipe, the
licenses of what the packages hold, the images of the README. The packages themselves are built and checked by the
build machines of GitHub (.github/workflows/build.yml, app/self_check.py).

Skipped if PySide6 is not installed (pip install -r requirements-app.txt).
"""
import os
from pathlib import Path
import re
import struct

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # before the application is created

from PIL import Image                               # noqa: E402
from PySide6.QtGui import QImage                    # noqa: E402

from app.i18n import LOCALES                        # noqa: E402
from app.icon import draw                           # noqa: E402
from app.theme import FONTS                         # noqa: E402
from release import images, licenses               # noqa: E402
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


def test_images_of_the_readme(tmp_path):
    """The three images, small and short: the window side by side, the robot alone, the GIF of a catch."""
    images.side_by_side(tmp_path / "side.png", size=(1200, 600), after=0.1)
    images.robot(tmp_path / "robot.png", size=(320, 180))
    images.catch(tmp_path / "catch.gif", size=(700, 420), seconds=0.4, fps=5)
    side, robot, catch = (Image.open(tmp_path / name) for name in ("side.png", "robot.png", "catch.gif"))
    assert side.size == (1200, 600) and robot.size == (320, 180)
    assert catch.format == "GIF" and catch.n_frames == 2 and catch.info["duration"] == 200
    for image in (side, robot.convert("RGB")):
        assert max(high - low for low, high in image.convert("RGB").getextrema()) > 100     # not uniform


def test_third_party_licenses():
    """Python first, the distributions of the modules given (app: SAPIR itself, none), Inter last; each with its
    license texts: Qt for Python with the LGPL and GPL version 3, which its wheels do not ship."""
    parts = licenses.sections(["PySide6", "mujoco", "numpy", "app"])
    titles = [title for title, _, _, _ in parts]
    assert titles[0].startswith("Python ") and titles[-1] == "Inter (font)"
    assert [title.split()[0] for title in titles[1:-1]] == ["mujoco", "numpy", "PySide6_Essentials"]
    assert all(files for _, _, files, _ in parts)
    text = licenses.text(["PySide6", "mujoco", "numpy", "app"])
    for words in ("PYTHON SOFTWARE FOUNDATION LICENSE", "Apache License", "GNU LESSER GENERAL PUBLIC LICENSE",
                  "GNU GENERAL PUBLIC LICENSE", "SIL OPEN FONT LICENSE", licenses.COPYRIGHT):
        assert words in text


def test_links_of_the_documentation():
    """The links and images of the README and of the technical documentation lead to files of the repository; the
    downloads, to the files that the build publishes with every version."""
    for page in (ROOT / "README.md", ROOT / "docs" / "technical.md"):
        for target in re.findall(r"\]\(([^)\s]+)\)", page.read_text(encoding="utf-8")):
            if not target.startswith(("http://", "https://", "#")):
                assert (page.parent / target).is_file(), (page.name, target)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    for name in ("SAPIR.exe", "SAPIR.dmg"):
        assert f"releases/latest/download/{name})" in readme and f"files/{name}" in workflow
