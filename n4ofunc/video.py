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
from string import Formatter
from typing import Any, Dict, List, Literal, Optional, Union, cast

import vapoursynth as vs
from fvsfunc import Depth
from vsutil import is_image, split

from .utils import (
    ColorRange,
    FieldBased,
    Matrix,
    Primaries,
    Transfer,
    has_plugin_or_raise,
    is_extension,
    snakeify,
)

__all__ = (
    "source",
    "src",
    "SimpleFrameReplace",
    "frame_replace",
    "sfr",
    "debug_clip",
    "debugclip",
    "shift_444",
)
core = vs.core
DitherType = Literal["none", "ordered", "random", "error_diffusion"]


class VideoSource:
    def __init__(self, path: Union[str, bytes, Path]):
        if isinstance(path, str):
            pass
        elif isinstance(path, Path):
            path = str(path)
        elif isinstance(src, bytes):
            path = path.decode("utf-8")
        else:
            raise TypeError("src must be a string or pathlib.Path")
        self.path = path

    def _open_source(self, force_lsmas: bool = False, **index_kwargs: Dict[str, Any]) -> vs.VideoNode:
        if is_extension(self.path, ".m2ts") or is_extension(self.path, ".mts"):
            force_lsmas = True
        if is_extension(self.path, ".d2v"):
            has_plugin_or_raise("d2v")
            return core.d2v.Source(self.path, **index_kwargs)
        elif is_extension(self.path, ".dgi"):
            has_plugin_or_raise("dgdecodenv")
            return core.dgdecodenv.DGSource(self.path, **index_kwargs)
        elif is_extension(self.path, ".mpls"):
            has_plugin_or_raise(["lsmas", "mpls"])
            pl_index = cast(int, index_kwargs.pop("playlist", 0))
            angle_index = cast(int, index_kwargs.pop("angle", 0))
            mpls_in = core.mpls.Read(self.path, pl_index, angle_index)

            clips: List[vs.VideoNode] = []
            for idx in range(mpls_in["count"]):  # type: ignore
                clips.append(core.lsmas.LWLibavSource(mpls_in["clip"][idx], **index_kwargs))  # type: ignore
            return core.std.Splice(clips)
        elif is_extension(self.path, ".avi"):
            has_plugin_or_raise("avisource")
            return core.avisource.AVISource(self.path, **index_kwargs)
        elif is_image(self.path):
            has_plugin_or_raise("imwri")
            return core.imwri.Read(self.path, **index_kwargs)
        if force_lsmas:
            has_plugin_or_raise("lsmas")
            return core.lsmas.LWLibavSource(self.path, **index_kwargs)
        has_plugin_or_raise("ffms2")
        return core.ffms2.Source(self.path, **index_kwargs)

    def open(self, force_lsmas: bool = False, **index_kwargs: Dict[str, Any]) -> vs.VideoNode:
        return self._open_source(force_lsmas, **index_kwargs)


def _map_x_to_yuv(bits: int):
    if bits >= 8 and bits < 9:
        return vs.YUV420P8
    elif bits >= 9 and bits < 10:
        return vs.YUV420P9
    elif bits >= 10 and bits < 12:
        return vs.YUV420P10
    elif bits >= 12 and bits < 14:
        return vs.YUV420P12
    elif bits >= 14 and bits < 16:
        return vs.YUV420P14
    elif bits >= 16 and bits < 18:
        return vs.YUV420P16
    else:
        return vs.YUV444PS


def source(
    src: Union[str, bytes, Path],
    lsmas: bool = False,
    depth: Optional[int] = None,
    dither_yuv: bool = True,
    matrix_s: str = "709",
    crop_l: int = 0,
    crop_r: int = 0,
    crop_t: int = 0,
    crop_b: int = 0,
    trim_start: Optional[int] = None,
    trim_end: Optional[int] = None,
    **index_kwargs: Dict[str, Any],
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
    matrix_s: :class:`str`
        The color matrix of the source.
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
    index_kwargs: :class:`Dict[str, Any]`
        The keyword arguments for the indexer.

    Returns
    -------
    :class:`VideoNode`
        The opened source.
    """

    def _trim(clip: vs.VideoNode, start: Optional[int] = None, end: Optional[int] = None) -> vs.VideoNode:
        if start is not None and end is not None:
            return clip.std.Trim(start, end)
        elif start is not None:
            return clip.std.Trim(start)
        elif end is not None:
            return clip.std.Trim(0, end)
        return clip

    def _crop(
        clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0
    ) -> vs.VideoNode:
        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return clip
        return clip.std.Crop(left, right, top, bottom)

    def _dither_to_yuv(clip: vs.VideoNode, matrix_src: str) -> vs.VideoNode:
        return clip.resize.Point(format=_map_x_to_yuv(clip.format.bits_per_sample), matrix_s=matrix_src)

    def _depth(clip: vs.VideoNode, depth: int, dither_type: DitherType = "error_diffusion") -> vs.VideoNode:
        return Depth(clip, depth, dither_type)

    vsrc = VideoSource(src)
    clip = vsrc.open(lsmas, **index_kwargs)
    if dither_yuv and clip.format.color_family != vs.YUV:
        clip = _dither_to_yuv(clip, matrix_s)
    if depth is not None:
        clip = _depth(clip, depth)
    return _crop(_trim(clip, trim_start, trim_end), crop_l, crop_r, crop_t, crop_b)


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


_DEFAULT_DEBUG_FORMAT = (
    "Frame {n} of {total_abs} ({total})\n"
    "Picture Type: {fp_pict_type}\n"
    "Resolution: {width}/{height} ({ar})\n"
    "FPS: {fps_num}/{fps_den} ({fps_frac})"
)


def debug_clip(
    clip: vs.VideoNode, format: str = _DEFAULT_DEBUG_FORMAT, alignment: int = 7, *, text_mode: bool = False
):
    """
    A helper function to show frame information.

    Rewritten!

    Changes from original:
    - Customizable format parameter
    Defaults to DEFAULT_DEBUG_FORMAT

    Available variables:
    - {n} - Current frame number
    - {total} - Total frames in the clip
    - {total_abs} - Total absolute frames in the clip (total - 1)
    - {width} - Width of the clip
    - {height} - Height of the clip
    - {ar} - Aspect ratio of the clip
    - {fps_num} - Numerator of the clip's FPS
    - {fps_den} - Denominator of the clip's FPS
    - {fps_frac} - Fraction of the clip's FPS (fps_num / fps_den)

    Anything in FrameProps will be available as {fp_prop_name} without the leading underscore.
    For example:
    - _PictType -> {fp_pict_type}
    - _SARNum -> {fp_sar_num}
    - _ColorRange -> {fp_color_range}

    Some props will be coerced to a string if possible.
    Any missing props will be replaced with the same string as the variable.

    Parameters
    ----------
    clip: :class:`VideoNode`
        The video to be debugged.
    format: :class:`str`
        What extra info do you want to add after main data.
    alignment: :class:`int`
        The alignment of the text. (Follow VapourSynth's `text.Text`)
    text_mode: :class:`bool`
        Whether to use :meth:`text.Text` or not for the debug info.
        Will automatically fallback to this if you don't have the `sub` module.

    Returns
    -------
    :class:`VideoNode`
        The clip with debug information.
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("debug_clip: clip must be a clip")
    has_plugin_or_raise(["sub", "text"], True)

    def _calc(i: Union[int, float], n: int, x: int):
        return str(i * (n / x))

    # fmt: off
    POS_MAP = {
        1: (15,   1070),
        2: (960,  1070),
        3: (1905, 1070),

        4: (15,   540),
        5: (960,  540),
        6: (1905, 540),

        7: (15,   10),
        8: (960,  10),
        9: (1905, 10),
    }
    # fmt: on

    def _generate_style(src_w: int, src_h: int, color: str = "00FFFF"):
        trg_w, trg_h = POS_MAP[alignment]
        gen = r"{\an" + str(alignment) + r"\b1\bord" + _calc(2.25, src_h, 1080) + r"\c&H" + color + r"\pos"
        gen += "({w}, {h})".format(w=_calc(trg_w, src_w, 1920), h=_calc(trg_h, src_h, 1080))
        gen += r"\fs" + _calc(24, src_h, 1080) + r"}"
        return gen

    use_text = not hasattr(core, "sub") or text_mode
    _text_style = _generate_style(clip.width, clip.height) if not use_text else ""

    class FormatDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    format = format.replace(r"\N", r"\n")

    def _add_frame_info(n: int, f: vs.VideoFrame, node: vs.VideoNode, fmt_text: str):
        # Unpack props
        total = node.num_frames
        total_abs = total - 1
        width, height = node.width, node.height
        res_ar = round(width / height, 4)
        fps_num = node.fps.numerator
        fps_den = node.fps.denominator
        fps_frac = round(fps_num / fps_den, 4)
        all_keys = list(f.props.keys())
        all_values = [f.props.get(k) for k in all_keys]
        all_keys_fmt = list(map(lambda x: "fp_" + snakeify(x[1:]), all_keys))
        format_dict = {
            "n": n,
            "total_abs": total_abs,
            "total": total,
            "width": width,
            "height": height,
            "ar": res_ar,
            "fps_num": fps_num,
            "fps_den": fps_den,
            "fps_frac": fps_frac,
        }
        format_dict.update(zip(all_keys_fmt, all_values))
        if "fp_color_range" in format_dict:
            format_dict["fp_color_range"] = ColorRange(format_dict["fp_color_range"]).as_str()
        if "fp_primaries" in format_dict:
            format_dict["fp_primaries"] = Primaries(format_dict["fp_primaries"]).as_str()
        if "fp_transfer" in format_dict:
            format_dict["fp_transfer"] = Transfer(format_dict["fp_transfer"]).as_str()
        if "fp_matrix" in format_dict:
            format_dict["fp_matrix"] = Matrix(format_dict["fp_matrix"]).as_str()
        if "fp_field_based" in format_dict:
            format_dict["fp_field_based"] = FieldBased(format_dict["fp_field_based"]).as_str()
        for k, v in format_dict.items():
            if isinstance(v, bytes):
                format_dict[k] = v.decode("utf-8", "replace")
        text_gen = _text_style
        formatter = Formatter()
        text_gen += formatter.vformat(fmt_text, (), FormatDict(format_dict))
        if use_text:
            node = node.text.Text(text_gen, alignment=alignment)
        else:
            node = node.sub.Subtitle(text_gen.replace(r"\n", r"\N"))
        return node[n]

    return core.std.FrameEval(clip, partial(_add_frame_info, node=clip, fmt_text=format), prop_src=clip)


def shift_444(clip: vs.VideoNode):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("shift_444: clip must be a clip")
    if clip.format.color_family != vs.YUV:
        raise TypeError("shift_444: clip must be YUV")
    Y, U, V = split(clip)
    U = core.resize.Point(U, clip.width, clip.height, src_left=0.5)
    V = core.resize.Point(V, clip.width, clip.height, src_left=0.5)
    return core.std.ShufflePlanes(clips=[Y, U, V], planes=[0, 0, 0], colorfamily=vs.YUV)


src = source
debugclip = debug_clip
sfr = SimpleFrameReplace
frame_replace = SimpleFrameReplace
