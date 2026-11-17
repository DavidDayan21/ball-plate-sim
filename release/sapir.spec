# PyInstaller recipe of the packages of SAPIR, from the root of the project:
#     python -m PyInstaller --noconfirm release/sapir.spec      (pip install -r requirements-build.txt)
# Windows: dist/SAPIR.exe, a single file (it unpacks itself in a temporary folder at every launch).
# Mac (Apple Silicon): dist/SAPIR.app, put in a .dmg by .github/workflows/build.yml.
# Both come with dist/THIRD-PARTY-LICENSES.txt (release/licenses.py), also inside them.
# Then: dist/SAPIR.exe --self-check report.txt (app/self_check.py) checks that nothing is missing.
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent                      # noqa: F821 (given by PyInstaller)
sys.path.insert(0, str(ROOT))
from app import NAME, VERSION                               # noqa: E402
from release.icons import write as write_icons              # noqa: E402
from release.licenses import write as write_licenses        # noqa: E402

ICONS = write_icons(Path(workpath) / "icons")               # noqa: F821
BUNDLE_ID = "io.github.daviddayan21.sapir"                  # identity of the Mac app

# Files read by the software, in the folders where it looks for them (next to its modules)
datas = [(str(ROOT / "app" / "fonts"), "app/fonts"),        # Inter and its license
         (str(ROOT / "app" / "locales"), "app/locales"),    # the texts
         (str(ROOT / "brains"), "brains")]                  # the AI brain
# Libraries loaded by name, out of reach of the analysis: MuJoCo (mujoco.dll on Windows; its plugins, unused) and
# GLFW, which gives MuJoCo its OpenGL context on Windows (on Mac: CGL, part of the system)
binaries = collect_dynamic_libs("mujoco") + collect_dynamic_libs("glfw")

a = Analysis([str(ROOT / "app" / "__main__.py")],           # noqa: F821
             pathex=[str(ROOT)], binaries=binaries, datas=datas,
             # OpenCV: only for the virtual camera, which the software does not use (synthetic sensor); PyOpenGL: only
             # for the OpenGL of MuJoCo without a screen, under Linux (EGL, OSMesa); Tk: unused
             excludes=["cv2", "OpenGL", "tkinter"])
# The licenses of what the analysis put in the package: modules, libraries and data, by their first folder
top_levels = ({name.split(".")[0] for name, _, _ in a.pure}
              | {Path(name).parts[0] for name, _, _ in a.binaries + a.datas})
LICENSES = write_licenses(Path(workpath) / "THIRD-PARTY-LICENSES.txt", top_levels)   # noqa: F821
a.datas += [(LICENSES.name, str(LICENSES), "DATA")]
pyz = PYZ(a.pure)                                           # noqa: F821

if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False, upx=False,   # noqa: F821
              target_arch="arm64", icon=str(ICONS["icns"]))
    collected = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)                       # noqa: F821
    app = BUNDLE(collected, name=f"{NAME}.app", icon=str(ICONS["icns"]), bundle_identifier=BUNDLE_ID,  # noqa: F821
                 version=VERSION, info_plist={"CFBundleDisplayName": NAME, "NSHighResolutionCapable": True,
                                              "LSApplicationCategoryType": "public.app-category.education"})
elif sys.platform == "win32":
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)
    numbers = tuple(int(n) for n in VERSION.split(".")) + (0,)
    texts = [StringStruct("ProductName", NAME), StringStruct("FileDescription", f"{NAME}: balancing robot"),
             StringStruct("ProductVersion", VERSION), StringStruct("FileVersion", VERSION),
             StringStruct("InternalName", NAME), StringStruct("OriginalFilename", f"{NAME}.exe")]
    version = VSVersionInfo(ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),     # properties of the file
                            kids=[StringFileInfo([StringTable("040904B0", texts)]),       # US English, Unicode
                                  VarFileInfo([VarStruct("Translation", [0x0409, 1200])])])
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=NAME, console=False, upx=False,   # noqa: F821
              icon=str(ICONS["ico"]), version=version)
else:                       # Linux: a single file too, only to try the recipe (no package is published for it)
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=NAME, console=False, upx=False)   # noqa: F821

shutil.copy2(LICENSES, Path(DISTPATH) / LICENSES.name)    # noqa: F821 (published with the packages)
