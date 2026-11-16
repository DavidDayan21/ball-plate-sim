"""The check of the software as packaged (app/self_check.py): python -m app --self-check [FILE]. Here on the
software as installed for development; the build machines of GitHub run it on the packages (release/).

Skipped if PySide6 is not installed (pip install -r requirements-app.txt).
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # before the application is created

import mujoco                                               # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402

from app import NAME, VERSION, i18n, self_check             # noqa: E402
from app.__main__ import main                               # noqa: E402


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def results(report) -> tuple:
    """({name of the check: its first word}, last line) of a report."""
    lines = report.read_text(encoding="utf-8").splitlines()
    return {line[9:].split(":")[0]: line.split()[0] for line in lines[:-1]}, lines[-1]


def test_self_check_passes(application, tmp_path, capsys):
    report = tmp_path / "report.txt"
    assert main(["SAPIR", "--self-check", str(report)]) == 0
    checks, verdict = results(report)
    assert checks == {name: "ok" for name, _ in self_check.CHECKS}
    assert verdict == f"{NAME} {VERSION} self-check: passed"
    assert capsys.readouterr().out == report.read_text(encoding="utf-8")       # also on the output
    assert main(["SAPIR", "--self-check"]) == 0                                 # no file: the output only
    assert capsys.readouterr().out.endswith("passed\n")


def test_missing_file_fails(application, tmp_path, monkeypatch):
    """A file left out of the package: its check fails, and those that need it; the others still run."""
    monkeypatch.setattr(i18n, "LOCALES", tmp_path / "nowhere")
    report = tmp_path / "report.txt"
    assert self_check.run(report) == 1
    checks, verdict = results(report)
    assert checks == dict(texts="FAILED", fonts="ok", brains="ok", OpenGL="ok", **{"robot shown": "ok"},
                          window="FAILED", drawing="FAILED")
    assert "FileNotFoundError" in report.read_text(encoding="utf-8") and verdict.endswith("failed")


def test_without_opengl_everything_else_checked(application, tmp_path, monkeypatch):
    """A computer that cannot draw the 3D (no usable OpenGL, as without a graphics card): the drawing is skipped,
    the rest checked, exit code 2."""
    def no_opengl(*args, **kwargs):
        raise mujoco.FatalError("an OpenGL platform library has not been loaded into this process")
    monkeypatch.setattr(mujoco, "Renderer", no_opengl)
    report = tmp_path / "report.txt"
    assert self_check.run(report) == 2
    checks, verdict = results(report)
    assert checks == {name: "skipped" if name == "drawing" else "ok" for name, _ in self_check.CHECKS}
    assert verdict.endswith("passed, without the drawing")
