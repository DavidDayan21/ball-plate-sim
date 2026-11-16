"""The icon files of the packages, drawn by app/icon.py at each size: SAPIR.ico for Windows (16 to 256 pixels, the
most the format holds) and SAPIR.icns for Mac (16 to 1024 pixels, Retina screens included), both made of PNG images.
Written in the build folder by release/sapir.spec (not kept in git: they are drawn again at every build).
"""
from pathlib import Path
import struct

from PySide6.QtCore import QBuffer, QIODevice

from app import NAME
from app.icon import SIZES, draw

ICO_SIZES = tuple(size for size in SIZES if size <= 256)
# (type, pixels) of the images of a .icns: 16, 32, 64, 128, 256 and 512 points, at 1× and 2× (Retina)
ICNS_TYPES = (("icp4", 16), ("ic11", 32), ("icp5", 32), ("ic12", 64), ("icp6", 64), ("ic07", 128), ("ic13", 256),
              ("ic08", 256), ("ic14", 512), ("ic09", 512), ("ic10", 1024))


def png(size: int) -> bytes:
    """The icon, size × size pixels, as a PNG file."""
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    if not draw(size).save(buffer, "PNG"):
        raise RuntimeError(f"PNG of {size} pixels not written")
    return bytes(buffer.data())


def ico() -> bytes:
    """Header, one entry per image (width and height 0 for 256), then the PNG images. Little endian."""
    images = [png(size) for size in ICO_SIZES]
    offset = 6 + 16 * len(images)
    entries = []
    for size, data in zip(ICO_SIZES, images):
        entries.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    return struct.pack("<HHH", 0, 1, len(images)) + b"".join(entries) + b"".join(images)


def icns() -> bytes:
    """"icns" and the length of the file, then per image its type, its length (8 bytes of header included) and its
    PNG. Big endian."""
    images = {size: png(size) for size in {size for _, size in ICNS_TYPES}}
    body = b"".join(kind.encode("ascii") + struct.pack(">I", 8 + len(images[size])) + images[size]
                    for kind, size in ICNS_TYPES)
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def write(folder) -> dict:
    """Writes SAPIR.ico and SAPIR.icns in `folder`: {"ico": path, "icns": path}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {"ico": folder / f"{NAME}.ico", "icns": folder / f"{NAME}.icns"}
    files["ico"].write_bytes(ico())
    files["icns"].write_bytes(icns())
    return files
