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
from typing import TYPE_CHECKING, List, Optional, Union

from vapoursynth import core

if TYPE_CHECKING:
    from vapoursynth import VideoFormat, VideoNode

__all__ = (
    "is_extension",
    "register_format",
    "has_plugin_or_raise",
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
        0 if yuv444 else clip.format.subsampling_h
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


def has_plugin_or_raise(plugins: Union[str, List[str]]):
    """
    Check if plugin exist in VapourSynth.
    If not raise, otherwise return ``True``.

    Parameters
    ----------
    plugins: :class:`str` or :class:`List[str]`
        The plugin name or a list of plugin names.

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
    for plugin in plugins:
        if not hasattr(core, plugin):
            raise RuntimeError(f"{caller_function}'{plugin}' is not installed or available in plugin list.")
    return True
