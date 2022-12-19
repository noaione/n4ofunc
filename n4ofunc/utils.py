"""
MIT License

Copyright (c) 2020-present noaione

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import inspect
import re
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

from vapoursynth import core

if TYPE_CHECKING:
    from vapoursynth import VideoFormat, VideoNode

__all__ = (
    "is_extension",
    "register_format",
    "has_plugin_or_raise",
    "snakeify",
)


def is_extension(filename: str, extension: str) -> bool:
    """
    Return ``True`` if filename has the same extension
    as the target extension.

    Parameters
    ----------
    filename: :class:`str`
        The filename that you want to compare with.
    extension: :class:`str`
        The target extension that you want to compare with.

    Returns
    --------
    :class:`bool`
        ``True`` if filename has the same extension as target.
    """
    filename = filename.lower()
    if extension.startswith("."):
        extension = extension[1:]
    dot_after = filename.rfind(".")
    if dot_after == -1:
        return False
    return filename[dot_after:] == extension


def register_format(clip: VideoNode, yuv444: bool = False) -> VideoFormat:
    """
    Create a new :class:`VideoFormat` object from a :class:`VideoNode`.

    Internally wraps :func:`vapoursynth.core.query_video_format` with
    fallbacks to :func:`vapoursynth.core.register_format` if the former
    is not available.

    Parameters
    ----------
    clip: :class:`VideoNode`
        The :class:`VideoNode` that you want to create a :class:`VideoFormat`
        object from.
    yuv444: :class:`bool`
        Do you want to register it as 4:4:4 subsampling or not

    Returns
    --------
    :class:`VideoFormat`
        The new :class:`VideoFormat` object.
    """
    # Check if query_video_format is available
    if not hasattr(core, "query_video_format"):
        # Fallback to register_format
        return core.register_format(
            clip.format.color_family,
            clip.format.sample_type,
            clip.format.bits_per_sample,
            0 if yuv444 else clip.format.subsampling_w,
            0 if yuv444 else clip.format.subsampling_h,
        )
    return core.query_video_format(
        clip.format.color_family,
        clip.format.sample_type,
        clip.format.bits_per_sample,
        0 if yuv444 else clip.format.subsampling_w,
        0 if yuv444 else clip.format.subsampling_h,
    )


def try_import(module: str, name: str) -> Optional[object]:
    """
    Try to import a module and return the imported object.

    Parameters
    ----------
    module: :class:`str`
        The module name.
    name: :class:`str`
        The object name.

    Returns
    --------
    :class:`Optional[object]`
        The imported object or ``None`` if it failed.
    """
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except ImportError:
        return None


def has_plugin_or_raise(plugins: Union[str, List[str]], only_once: bool = False):
    """
    Check if plugin exist in VapourSynth.
    If not raise, otherwise return ``True``.

    Parameters
    ----------
    plugins: :class:`str` or :class:`List[str]`
        The plugin name or a list of plugin names.
    only_once: :class:`bool`
        If ``True``, only raise if all plugins are not found.

    Returns
    --------
    :class:`bool`
        ``True`` if the library is available.

    Raises
    --------
    :class:`RuntimeError`
        If the library is not available.
    """
    try:
        caller_function = inspect.stack()[1].function + ": "
    except Exception:
        caller_function = ""
    if isinstance(plugins, str):
        plugins = [plugins]
    any_found = False
    for plugin in plugins:
        if hasattr(core, plugin):
            any_found = True
            if only_once:
                break
    if not any_found:
        raise RuntimeError(f"{caller_function}'{plugin}' is not installed or available in plugin list.")
    return True


class Primaries(int, Enum):
    BT709 = 1
    UNKNOWN = 2
    BT470M = 4
    BT470BG = 5
    ST170M = 6
    ST240M = 7
    FILM = 8
    BT2020 = 9
    ST428 = 10
    XYZ = ST428
    ST431_2 = 11
    ST432_1 = 12
    EBU3213E = 22

    def as_str(self) -> str:
        return self.name.upper().replace("_", " ")


class Matrix(int, Enum):
    RGB = 0
    GBR = RGB
    BT709 = 1
    UNKNOWN = 2
    FCC = 4
    BT470BG = 5
    BT601 = BT470BG
    SMPTE170M = 6
    SMPTE240M = 7
    BT2020NC = 9
    BT2020C = 10
    SMPTE2085 = 11
    CHROMA_DERIVED_NC = 12
    CHROMA_DERIVED_C = 13
    ICTCP = 14

    def as_str(self) -> str:
        return self.name.upper().replace("_", " ")


class Transfer(int, Enum):
    BT709 = 1
    UNKNOWN = 2
    BT470M = 4
    BT470BG = 5
    BT601 = 6
    ST240M = 7
    LINEAR = 8
    LOG_100 = 9
    LOG_316 = 10
    XVYCC = 11
    SRGB = 13
    BT2020_10bits = 14
    BT2020_12bits = 15
    ST2084 = 16
    ARIB_B67 = 18

    # libplacebo Transfer
    # Standard gamut:
    BT601_525 = 100
    BT601_625 = 101
    EBU_3213 = 102
    # Wide gamut:
    APPLE = 103
    ADOBE = 104
    PRO_PHOTO = 105
    CIE_1931 = 106
    DCI_P3 = 107
    DISPLAY_P3 = 108
    V_GAMUT = 109
    S_GAMUT = 110
    FILM_C = 111

    def as_str(self) -> str:
        return self.name.upper().replace("_", " ")


class ColorRange(int, Enum):
    LIMITED = 0
    FULL = 1

    def as_str(self) -> str:
        return "Limited" if self == ColorRange.LIMITED else "Full"


class FieldBased(int, Enum):
    PROGRESSIVE = 0
    BOTTOM_FIELD_FIRST = 1
    TOP_FIELD_FIRST = 2

    def as_str(self) -> str:
        lstr = self.name.lower().split("_")
        return " ".join([s.capitalize() for s in lstr])


def snakeify(word: str):
    # https://github.com/jpvanhal/inflection/blob/master/inflection/__init__.py#L397
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
    word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
    word = word.replace("-", "_")
    return word.lower()
