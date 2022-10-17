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

import warnings
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import fvsfunc as fvf
import vapoursynth as vs
from vsutil import get_depth, get_w, get_y, iterate

from .mask import simple_native_mask
from .utils import has_plugin_or_raise, register_format

__all__ = (
    "masked_descale",
    "upscale_nnedi3",
    "adaptive_scaling",
    "adaptive_rescale",
    "adaptive_descale",
)
core = vs.core
IntegerFloat = Union[int, float]
VALID_KERNELS = ["bicubic", "bilinear", "lanczos", "spline16", "spline36", "spline64"]
DescaleKernel = Literal["bicubic", "bilinear", "lanczos", "spline16", "spline36", "spline64"]
VALID_RESCALE_KERNEL = [
    "point",
    "rect",
    "linear",
    "cubic",
    "lanczos",
    "blackman",
    "blackmanminlobe",
    "spline16",
    "spline36",
    "spline64",
    "spline",
    "gauss",
    "sinc",
    "impulse",
]
RescaleKernel = Literal[
    "point",
    "rect",
    "linear",
    "cubic",
    "lanczos",
    "blackman",
    "blackmanminlobe",
    "spline16",
    "spline36",
    "spline64",
    "spline",
    "gauss",
    "sinc",
    "impulse",
]


def _descale_video(
    src: vs.VideoNode,
    width: int,
    height: int,
    kernel: DescaleKernel = "bicubic",
    b: IntegerFloat = 1 / 3,
    c: IntegerFloat = 1 / 3,
    taps: int = 3,
    yuv444: bool = False,
):
    """A modified version of descale.py"""

    has_plugin_or_raise("descale")
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h

    if src_cf == vs.RGB:
        rgb = src.resize.Point(format=vs.RGBS).descale.Descale(
            width,
            height,
            kernel=kernel,
            taps=taps,
            b=b,
            c=c,
        )
        return rgb.resize.Point(format=src_f.id)

    y = src.resize.Point(format=vs.GRAYS).descale.Descale(width, height, kernel, taps=taps, b=b, c=c)
    y_f = core.register_format(vs.GRAY, src_st, src_bits, 0, 0)
    y = y.resize.Point(format=y_f.id)

    if src_cf == vs.GRAY:
        return y

    if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
        raise ValueError("descale: The output dimension and the subsampling are incompatible.")

    uv_f = register_format(src, yuv444)
    uv = src.resize.Spline36(width, height, format=uv_f.id)
    return core.std.ShufflePlanes([y, uv], [0, 1, 2], vs.YUV)


def _get_resizer(b: IntegerFloat, c: IntegerFloat, taps: int, kernel: DescaleKernel):
    if kernel == "bilinear":
        return core.resize.Bilinear
    elif kernel == "bicubic":
        return partial(core.resize.Bicubic, filter_param_a=b, filter_param_b=c)
    elif kernel == "lanczos":
        return partial(core.resize.Lanczos, filter_param_a=taps)
    elif kernel == "spline16":
        return core.resize.Spline16
    elif kernel == "spline36":
        return core.resize.Spline36
    elif kernel == "spline64":
        return core.resize.Spline64
    raise ValueError(f"masked_descale: Invalid kernel: {kernel}")


def _square_clip(clip: vs.VideoNode):
    clip_len = len(clip)
    top = core.std.BlankClip(length=clip_len, format=vs.GRAYS, height=4, width=10, color=[1])
    side = core.std.BlankClip(length=clip_len, format=vs.GRAYS, height=2, width=4, color=[1])
    center = core.std.BlankClip(length=clip_len, format=vs.GRAYS, height=2, width=2, color=[0])
    h_stack = core.std.StackHorizontal([side, center, side])
    return core.std.StackVertical([top, h_stack, top])


def _kirsch(src: vs.VideoNode) -> vs.VideoNode:
    """
    Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
    more information: https://ddl.kageru.moe/konOJ.pdf

    From kagefunc.py, moved here so I don't need to import it.
    """
    kirsch1 = src.std.Convolution(
        matrix=[5, 5, 5, -3, 0, -3, -3, -3, -3], saturate=False
    )  # noqa: E241,E201,E501
    kirsch2 = src.std.Convolution(matrix=[-3, 5, 5, -3, 0, 5, -3, -3, -3], saturate=False)  # noqa: E241
    kirsch3 = src.std.Convolution(matrix=[-3, -3, 5, -3, 0, 5, -3, -3, 5], saturate=False)  # noqa: E241
    kirsch4 = src.std.Convolution(matrix=[-3, -3, -3, -3, 0, 5, -3, 5, 5], saturate=False)  # noqa: E241
    return core.std.Expr([kirsch1, kirsch2, kirsch3, kirsch4], "x y max z max a max")


def _retinex_edgemask(src: vs.VideoNode, sigma: int = 1) -> vs.VideoNode:
    """
    Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
    sigma is the sigma of tcanny

    From kagefunc.py, moved here so I don't need to import it.
    """
    luma = get_y(src)
    max_value = 1 if src.format.sample_type == vs.FLOAT else (1 << get_depth(src)) - 1
    ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    tcanny = ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    return core.std.Expr([_kirsch(luma), tcanny], f"x y + {max_value} min")


def masked_descale(
    src: vs.VideoNode,
    target_w: IntegerFloat,
    target_h: IntegerFloat,
    kernel: DescaleKernel = "bicubic",
    resize_kernel: Optional[DescaleKernel] = None,
    b: IntegerFloat = 1 / 3,
    c: IntegerFloat = 1 / 3,
    taps: int = 3,
    yuv444: bool = False,
    expandN: int = 1,
    masked: bool = True,
    show_mask: bool = False,
) -> vs.VideoNode:
    """
    Descale a video with a native mask.

    Valid kernel:
    - bicubic
    - bilinear
    - lanczos
    - spline16
    - spline36
    - spline64

    Default to ``bicubic`` kernel.

    Parameters
    ----------
    src: :class:`VideoNode`
        The video source.
    target_w: :class:`Union[int, float]`
        Target descale width.
    target_h: :class:`Union[int, float]`
        Target descale height.
    kernel: :class:`DescaleKernel`
        Kernel used for descaling.
    resize_kernel: :class:`DescaleKernel`
        Kernel used for resizing, if None, will use the same as kernel as descaler.
    b: :class:`Union[int, float]`
        B-parameter for the kernel.
    c: :class:`Union[int, float]`
        C-parameter for the kernel.
    taps: :class:`int`
        Lanczos taps for the kernel.
    yuv444: :class:`bool`
        Dither to 4:4:4 chroma subsampling.
    expandN: :class:`int`
        Iteration count that will expand the mask size.
    masked: :class:`bool`
        Do you want to apply the mask or not.
    show_mask: :class:`bool`
        Do you want to show the mask or not.

    Returns
    -------
    :class:`VideoNode`
        The descaled video.
    """
    if not isinstance(src, vs.VideoNode):
        raise TypeError("masked_descale: The source must be a clip.")

    if expandN is None:
        expandN = -1
    if expandN < 0:
        raise ValueError("masked_descale: expandN cannot be negative integer")

    kernel = kernel.lower()  # type: ignore
    if kernel not in VALID_KERNELS:
        raise ValueError(f"masked_descale: Invalid kernel: {kernel}")

    target_w = int(round(target_w))
    target_h = int(round(target_h))

    descale = _descale_video(src, target_w, target_h, kernel, b, c, taps, yuv444)
    des_f = register_format(descale, yuv444)
    Resizer = _get_resizer(b, c, taps, resize_kernel or kernel)
    nr_resize = Resizer(src, target_w, target_h, format=des_f.id)

    if masked:
        video_mask = simple_native_mask(src, target_w, target_h, iter_max=expandN)
        if show_mask:
            return video_mask
        return core.std.MaskedMerge(descale, nr_resize, video_mask)
    return descale


def upscale_nnedi3(
    src: vs.VideoNode,
    rfactor: int = 2,
    width: Optional[int] = None,
    height: Optional[int] = None,
    correct_shift: bool = True,
    kernel: Optional[RescaleKernel] = "spline36",
    use_gpu: bool = False,
    **nnedi3_kwargs: Dict[str, Any],
) -> vs.VideoNode:
    """
    A modern port of nnedi3_rpow2

    Parameters
    -----------
    src: :class:`VideoNode`
        The video source.
    rfactor: :class:`int`
        Image enlargement factor. Muse be a power of 2 in range of [2 to 1024].
    width: :class:`int`
        If correcting the image center shift by using the correct_shift parameter,
        this parameter is used to specify the width of the image.
    height: :class:`int`
        If correcting the image center shift by using the correct_shift parameter,
        this parameter is used to specify the height of the image.
    correct_shift: :class:`bool`
        Correct shift if ``True``. The correction is accomplished by using
        the subpixel cropping capability of fmtc's resizers.
    kernel: :class:`RescaleKernel`
        Sets the resizer used for correcting the image center shift that rescaler
        introduces. This can be any fmtc's kernels, such as "cubic", "spline36", etc.
    use_gpu: :class:`bool`
        Use GPU for rescaling (need nnedi3cl).
    nnedi3_kwargs: :class:`Dict[str, Any]`
        Keyword arguments for nnedi3. Refer to nnedi3 documentation.

    Returns
    -------
    :class:`VideoNode`
        The upscaled video.
    """

    if width is None:
        width = src.width * rfactor
    if height is None:
        height = src.height * rfactor

    if kernel is None:
        kernel = "spline36"
    kernel = kernel.lower()  # type: ignore
    if kernel not in VALID_RESCALE_KERNEL:
        raise ValueError(f"upscale_nnedi3: Invalid kernel: {kernel}")

    hshift = 0.0
    vshift = -0.5
    nnedi3_kwargs.pop("dh", None)
    pkdnnedi = {
        "dh": True,
        **nnedi3_kwargs,
    }
    pkdchroma = {
        "kernel": kernel,
        "sy": -0.5,
        "planes": [2, 3, 3],
    }

    tmp = 1
    times = 0
    while tmp < rfactor:
        tmp *= 2
        times += 1

    if rfactor < 2 or rfactor > 1024:
        raise ValueError("rescale: rfactor must be between 2 and 1024")
    if tmp != rfactor:
        raise ValueError("rescale: rfactor must be a power of 2")
    if not hasattr(core, "nnedi3"):
        raise RuntimeError("rescale: nnedi3 plugin is required")

    if correct_shift or src.format.subsampling_h:
        if not hasattr(core, "fmtc"):
            raise RuntimeError("rescale: fmtconv plugin is required")
    has_cl = hasattr(core, "nnedi3cl")
    if use_gpu and not has_cl:
        warnings.warn("rescale: nnedi3cl plugin is required since use_gpu is True", RuntimeWarning)

    last = src
    for i in range(times):
        field = 1 if i == 0 else 0
        if use_gpu and has_cl:
            last = core.nnedi3cl.NNEDI3CL(last, field=field, **pkdnnedi)
        else:
            last = core.nnedi3.nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)
        if last.format.subsampling_w:
            # Apparently always using field=1 for the horizontal pass somehow
            # keeps luma/chroma alignment.
            field = 1
            hshift = hshift * 2 - 0.5
        else:
            hshift = -0.5
        if use_gpu and has_cl:
            last = core.nnedi3cl.NNEDI3CL(last, field=field, **pkdnnedi)
        else:
            last = core.nnedi3.nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)

    # Correct vertical shift of the chroma.
    if src.format.subsampling_h:
        last = core.fmtc.resample(last, w=last.width, h=last.height, **pkdchroma)
    if correct_shift:
        last = core.fmtc.resample(last, w=width, h=height, kernel=cast(str, kernel), sx=hshift, sy=vshift)

    if last.format.id != src.format.id:
        last = core.fmtc.bitdepth(last, csp=src.format.id)
    return last


def adaptive_scaling(
    src: vs.VideoNode,
    target_w: Optional[IntegerFloat] = None,
    target_h: Optional[IntegerFloat] = None,
    descale_range: List[int] = [],
    kernel: DescaleKernel = "bicubic",
    b: IntegerFloat = 1 / 3,
    c: IntegerFloat = 1 / 3,
    taps: int = 3,
    iter_max: int = 3,
    rescale: bool = True,
    use_gpu: bool = False,
    show_native_res: bool = False,
    show_mask: bool = False,
):
    """
    Descale a video within the range and upscale it back
    to the target_w and target_h.

    If target is not defined, it will use original resolution.

    Written originally by kageru, modified by N4O.

    Valid kernel:
    - bicubic
    - bilinear
    - lanczos
    - spline16
    - spline36
    - spline64

    Default to ``bicubic`` kernel.

    Parameters
    ----------
    src: :class:`VideoNode`
        The video source.
    target_w: :class:`Union[int, float]`
        Target final video width.
    target_h: :class:`Union[int, float]`
        Target final video height.
    descale_range: :class:`List[Union[int, float]]`
        List of number of descale target height.
    kernel: :class:`DescaleKernel`
        Kernel used for descaling.
    b: :class:`Union[int, float]`
        B-parameter for the kernel.
    c: :class:`Union[int, float]`
        C-parameter for the kernel.
    taps: :class:`int`
        Lanczos taps for the kernel.
    iter_max: :class:`int`
        Iteration count that will expand the mask size.
    rescale: :class:`bool`
        Rescale the video if ``True``.
    use_gpu: :class:`bool`
        Use GPU for rescaling (need nnedi3cl).
    show_native_res: :class:`bool`
        Show a text notifying what the native resolution are.
    show_mask: :class:`bool`
        Do you want to show the mask or not.
    """
    target_w = src.width if target_w is None else target_w
    target_h = src.height if target_h is None else target_h
    kernel = kernel.lower()  # type: ignore
    if kernel not in VALID_KERNELS:
        raise ValueError(f"adaptive_scaling: Invalid kernel: {kernel}")
    if not isinstance(src, vs.VideoNode):
        raise TypeError("adaptive_scaling: The source must be a clip.")
    if not isinstance(descale_range, (list, tuple)):
        raise TypeError("adaptive_scaling: descale_range must be a list.")

    descale_range = list(descale_range)
    if len(descale_range) != 2:
        raise ValueError("adaptive_scaling: descale_range must have only 2 elements.")
    if descale_range[0] > descale_range[1]:
        raise ValueError("adaptive_scaling: descale_range first value cannot be larger than second value")

    if rescale:
        if descale_range[0] > target_h or descale_range[1] > target_h:
            raise ValueError(
                "adaptive_scaling: One of the descale_range value cannot be larger than target_h"
            )
    has_plugin_or_raise(["retinex", "tcanny", "descale", "fmtc", "rgvs", "nnedi3"])
    target_w = int(round(target_w))
    target_h = int(round(target_h))

    if target_w % 2 != 0:
        raise ValueError("adaptive_scaling: target_w must be even.")
    if target_h % 2 != 0:
        raise ValueError("adaptive_scaling: target_h must be even.")

    ref = src
    ref_d = ref.format.bits_per_sample
    clip32 = fvf.Depth(ref, 32)
    y = get_y(clip32)
    global_clip_resizer = _get_resizer(b, c, taps, kernel)
    global_clip_descaler = partial(core.descale.Descale, kernel=kernel, b=b, c=c, taps=taps)

    def simple_descale(y_clip: vs.VideoNode, h: int) -> Tuple[vs.VideoNode, vs.VideoNode]:
        ar = y_clip.width / y_clip.height
        down = global_clip_descaler(y_clip, get_w(h, ar), h)
        if rescale:
            up = global_clip_resizer(down, target_w, target_h)
        else:
            up = global_clip_resizer(down, y_clip.width, y_clip.height)
        diff = core.std.Expr([y, up], "x y - abs").std.PlaneStats()
        return down, diff

    descaled_clips_list = [simple_descale(y, h) for h in range(descale_range[0], descale_range[1])]
    descaled_clips = [clip[0] for clip in descaled_clips_list]
    descaled_props = [clip[1] for clip in descaled_clips_list]

    if not rescale:
        y = global_clip_resizer(y, target_w, target_h)
        clip32 = global_clip_resizer(clip32, target_w, target_h)

    def select_scale(n: int, f: List[vs.VideoFrame], descale_list: List[vs.VideoNode]):
        errors = [x.props["PlaneStatsAverage"] for x in f]
        y_deb = descale_list[errors.index(min(errors))]  # type: ignore
        dmask = core.std.Expr(
            [y, global_clip_resizer(y_deb, target_w, target_h)], "x y - abs 0.025 > 1 0 ?"
        ).std.Maximum()
        y_deb16 = fvf.Depth(y_deb, 16)

        if rescale:
            y_scaled = upscale_nnedi3(
                y_deb16, nns=4, correct_shift=True, width=target_w, height=target_h, use_gpu=use_gpu  # type: ignore
            ).fmtc.bitdepth(bits=32)
        else:
            y_scaled = global_clip_resizer(y_deb16, target_w, target_h).fmtc.bitdepth(bits=32)
        dmask = global_clip_resizer(dmask, target_w, target_h)
        if show_native_res and not show_mask:
            y_scaled = y_scaled.text.Text(f"Native resolution for this frame: {y_deb.height}")
        return core.std.ClipToProp(y_scaled, dmask)

    y_deb = core.std.FrameEval(y, partial(select_scale, descale_list=descaled_clips), prop_src=descaled_props)
    dmask = core.std.PropToClip(y_deb)

    line = core.std.StackHorizontal([_square_clip(y)] * (target_w // 10))
    full = core.std.StackVertical([line] * (target_h // 10))

    line_mask = global_clip_resizer(full, target_w, target_h)

    artifacts = core.misc.Hysteresis(
        global_clip_resizer(dmask, target_w, target_h, format=vs.GRAYS),
        core.std.Expr([get_y(clip32).tcanny.TCanny(sigma=3), line_mask], "x y min"),
    )

    ret_raw = _retinex_edgemask(ref)
    if not rescale:
        ret_raw = global_clip_resizer(ret_raw, target_w, target_h)

    ret = ret_raw.std.Binarize(30).rgvs.RemoveGrain(3)
    mask = core.std.Expr(
        [iterate(artifacts, core.std.Maximum, iter_max), ret.resize.Point(format=vs.GRAYS)], "y x -"
    ).std.Binarize(0.4)
    mask = mask.std.Inflate().std.Convolution([1] * 9).std.Convolution([1] * 9)

    if show_mask:
        return mask

    merged = core.std.MaskedMerge(y, y_deb, mask)
    merged = core.std.ShufflePlanes([merged, clip32], [0, 1, 2], vs.YUV)
    return fvf.Depth(merged, ref_d)


adaptive_rescale = partial(adaptive_scaling, rescale=True)
adaptive_descale = partial(adaptive_scaling, rescale=False)
