"""THIRD-PARTY-LICENSES.txt: the licenses of what the packages hold besides the code of SAPIR. Written by
release/sapir.spec in every package (and next to it in dist/, published with it): Python, then every library (Python
distribution) that a module or a library file of the package comes from, with the license files it ships, read where
it is installed; the LGPL and GPL version 3 for Qt and Qt for Python, whose wheels ship no license file
(release/licenses/: the texts of the FSF, copied from /usr/share/common-licenses of Debian); the font Inter.
"""
from importlib import metadata
from pathlib import Path
import platform
import sys
import sysconfig

from app import NAME, VERSION

ROOT = Path(__file__).resolve().parent.parent
TEXTS = Path(__file__).resolve().parent / "licenses"
WORDS = ("LICEN", "COPYING", "NOTICE")          # in the name of a license file
QT = ("pyside6-essentials", "shiboken6")        # distributions of Qt for Python (and Qt)
QT_PARTS = "https://doc.qt.io/qt-6/licenses-used-in-qt.html"
RULE = "=" * 100
COPYRIGHT = "© 2026 DavidDayan21, all rights reserved"     # the code of SAPIR: no license


def _normal(name: str) -> str:
    return name.lower().replace("_", "-")


def distributions(top_levels) -> list:
    """The distributions installed that provide these top-level modules or folders, each once, sorted by name."""
    owners = metadata.packages_distributions()
    names = {d for top in top_levels for d in owners.get(top, ())}
    return [metadata.distribution(n) for n in sorted({_normal(n) for n in names})]


def license_files(distribution) -> list:
    """The files of the distribution whose name says license, copying or notice (Path)."""
    files = [f for f in distribution.files or () if any(w in f.name.upper() for w in WORDS)]
    paths = [Path(distribution.locate_file(f)) for f in sorted(files, key=str)]
    return [path for path in paths if path.is_file()]


def license_name(distribution) -> str:
    """Its license as declared: expression, short license field, or classifiers."""
    m = distribution.metadata
    short = m.get("License") if m.get("License") and "\n" not in m["License"] and len(m["License"]) <= 80 else None
    classifiers = [c.split(" :: ")[-1] for c in m.get_all("Classifier") or () if c.startswith("License ::")]
    return m.get("License-Expression") or short or ", ".join(classifiers) or "see its license files"


def python_license() -> Path:
    for path in (Path(sys.base_prefix, "LICENSE.txt"), Path(sysconfig.get_paths()["stdlib"], "LICENSE.txt")):
        if path.is_file():
            return path
    raise FileNotFoundError("license of Python not found")


def sections(top_levels) -> list:
    """(title, license, [files], note) of every part, in the order of the file."""
    parts = [(f"Python {platform.python_version()}", "PSF-2.0", [python_license()], "")]
    for d in distributions(top_levels):
        title, files, note = f"{d.metadata['Name']} {d.version}", license_files(d), ""
        if _normal(d.metadata["Name"]) in QT:
            files = [TEXTS / "LGPL-3.0.txt", TEXTS / "GPL-3.0.txt"]
            note = (f"Used under the LGPL version 3 (whose terms include those of the GPL version 3). Qt itself holds "
                    f"parts under other licenses: {QT_PARTS}")
        elif not files:
            note = "This distribution ships no license file."
        parts.append((title, license_name(d), files, note))
    parts.append(("Inter (font)", "OFL-1.1", [ROOT / "app" / "fonts" / "OFL.txt"], ""))
    return parts


def text(top_levels) -> str:
    parts = sections(top_levels)
    lines = [f"{NAME} {VERSION}: third-party licenses", "",
             f"Besides the code of {NAME} ({COPYRIGHT}), its packages hold the software below,",
             "each under its own license. Contents:", ""]
    lines += [f"  {title} ({license})" for title, license, _, _ in parts]
    for title, license, files, note in parts:
        lines += ["", RULE, f"{title} ({license})", RULE]
        if note:
            lines += [note]
        for path in files:
            lines += ["", f"--- {path.name} ---", "", path.read_text(encoding="utf-8", errors="replace").rstrip()]
    return "\n".join(lines) + "\n"


def write(path, top_levels) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text(top_levels), encoding="utf-8")
    return path
