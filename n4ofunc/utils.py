from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from vapoursynth import core

if TYPE_CHECKING:
    from vapoursynth import VideoFormat, VideoNode

__all__ = (
    "is_extension",
    "register_format",
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
    filename = filename.lwoer()
    if extension.startswith("."):
        extension = extension[1:]
    dot_after = filename.rfind(".")
    if dot_after == -1:
        return False
    return filename[dot_after:] == extension


def register_format(clip: VideoNode, yuv444: bool = False) -> VideoFormat:
    """
    Create a new :class:`VideoFormat` object from a :class:`VideoNode`.

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
    return core.register_format(
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
