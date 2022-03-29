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

from functools import partial
from pathlib import Path
from typing import Optional, Union

import vapoursynth as vs
from fvsfunc import Depth
from vsutil import is_image

from .utils import is_extension

__all__ = (
    "source",
    "src",
    "SimpleFrameReplace",
    "frame_replace",
    "sfr",
    "debug_clip",
    "debugclip",
)
core = vs.core


def _source_func(src: str, force_lsmas: bool = False):
    if is_extension(src, ".m2ts") or is_extension(src, ".ts"):
        force_lsmas = True
    if is_extension(src, ".d2v"):
        return core.d2v.Source
    if is_extension(src, ".avi"):
        return core.avisource.AVISource
    if is_image(src):
        return core.imwri.Read
    if force_lsmas:
        return core.lsmas.LWLibavSource
    return core.ffms2.Source


def source(
    src: Union[str, bytes, Path],
    lsmas: bool = False,
    depth: Optional[int] = None,
    dither_yuv: bool = True,
    crop_l: int = 0,
    crop_r: int = 0,
    crop_t: int = 0,
    crop_b: int = 0,
    trim_start: Optional[int] = None,
    trim_end: Optional[int] = None,
) -> vs.VideoNode:
    """
    Open a video or image source.

    Parameters
    ----------
    src: :class:`Union[str, bytes, Path]`
        The source to be opened.
    lsmas: :class:`bool`
        Whether to force use LSMASHSource or not.
    depth: :class:`Optional[int]`
        Change the bitdepth of the source.
    dither_yuv: :class:`bool`
        Whether to dither to YUV source or not.
    crop_l: :class:`int`
        The left crop.
    crop_r: :class:`int`
        The right crop.
    crop_t: :class:`int`
        The top crop.
    crop_b: :class:`int`
        The bottom crop.
    trim_start: :class:`Optional[int]`
        The start frame to be trimmed.
    trim_end: :class:`Optional[int]`
        The end frame to be trimmed.

    Returns
    -------
    :class:`VideoNode`
        The opened source.
    """

    if isinstance(src, str):
        pass
    elif isinstance(src, Path):
        src = str(src)
    elif isinstance(src, bytes):
        src = src.decode("utf-8")
    else:
        raise TypeError("src must be a string or pathlib.Path")

    clip = _source_func(src, lsmas)(src)
    if dither_yuv and clip.format.color_family != vs.YUV:
        clip = clip.resize.Point(format=vs.YUV420P8, matrix_s="709")

    if isinstance(depth, int):
        clip = Depth(clip, depth)
    if trim_start is not None and trim_end is not None:
        clip = clip.std.Trim(trim_start, trim_end)
    elif trim_start is not None:
        clip = clip.std.Trim(trim_start)
    elif trim_end is not None:
        clip = clip.std.Trim(0, trim_end)
    if crop_l or crop_r or crop_b or crop_t:
        clip = clip.std.Crop(crop_l, crop_r, crop_t, crop_b)
    return clip


def SimpleFrameReplace(clip: vs.VideoNode, src_frame: int, target_frame: str):
    """
    Replace a clip target frame from selected source frame.

    Parameters
    ----------
    clip: :class:`VideoNode`
        The clip to be processed.
    src_frame: :class:`int`
        The source frame to be replaced.
    target_frame: :class:`int`
        The target frame to be replaced.

    Returns
    -------
    :class:`VideoNode`
        The processed clip.
    """

    clip_src = clip[src_frame]
    frame_range = target_frame.split("-")
    if len(frame_range) < 2:
        frame_range = [int(frame_range[0]), int(frame_range[0]) + 1]
    else:
        frame_range = [int(frame_range[0]), int(frame_range[1]) + 1]

    if frame_range[0] > frame_range[1]:
        raise ValueError("SimpleFrameReplace: `target_frame` last range number are bigger than the first one")

    clip_src = clip_src * (frame_range[1] - frame_range[0])
    pre = clip[: frame_range[0]]
    post = clip[frame_range[1] :]
    return pre + clip_src + post


def debug_clip(clip: vs.VideoNode, extra_info: Optional[str] = None):
    """
    A helper function to show frame information.

    Parameters
    ----------
    clip: :class:`VideoNode`
        The video to be debugged.
    extra_info: :class:`Optional[str]`
        What extra info do you want to add after main data.

    Returns
    -------
    :class:`VideoNode`
        The clip with debug information.
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("debug_clip: clip must be a clip")

    def _calc(i: int, n: int, x: int):
        return str(i * (n / x))

    def _generate_style(src_w: int, src_h: int, color: str = "00FFFF"):
        gen = r"{\an7\b1\bord" + _calc(2.25, src_h, 1080) + r"\c&H" + color + r"\pos"
        gen += "({w}, {h})".format(w=_calc(15, src_w, 1920), h=_calc(10, src_h, 1080))
        gen += r"\fs" + _calc(24, src_h, 1080) + r"}"
        return gen

    if extra_info is not None:
        extra_info = extra_info.replace(r"\n", r"\N")

    _main_style = _generate_style(clip.width, clip.height)
    _extra_style: Optional[str] = None
    if extra_info is not None:
        _extra_style = _generate_style(clip.width, clip.height, "FFFFFF")

    def _add_frame_info(n: int, f: vs.VideoFrame, node: vs.VideoNode):
        text_gen = _main_style
        # Frame
        text_gen += f"Frame {n} of {node.num_frames -1}\\N"
        # PictType
        text_gen += f"Picture Type: {f.props['_PictType'].decode()}\\N"
        # Resolution
        width, height = node.width, node.height
        res_ar = round(width / height, 4)
        text_gen += f"Resolution: {width}/{height} ({res_ar})\\N"
        # FPS
        fps_num, fps_den = node.fps.numerator, node.fps.denominator
        fps_ar = round(fps_num / fps_den, 4)
        text_gen += f"FPS: {fps_num}/{fps_den} ({fps_ar})"
        if extra_info is not None and _extra_style is not None:
            text_gen += f"\\N {_extra_style}{extra_info}"
        node = node.sub.Subtitle(text_gen)
        return node[n]

    return core.std.FrameEval(clip, partial(_add_frame_info, node=clip), prop_src=clip)


src = source
debugclip = debug_clip
sfr = SimpleFrameReplace
frame_replace = SimpleFrameReplace
