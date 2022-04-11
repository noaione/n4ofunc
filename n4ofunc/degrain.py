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

from functools import partial
from typing import Any, Callable, Dict, Literal

import vapoursynth as vs
from vsutil import get_y, iterate

from .utils import try_import

__all__ = (
    "adaptive_degrain2",
    "adaptive_bm3d",
    "adaptive_dfttest",
    "adaptive_knlm",
    "adaptive_tnlm",
    "adaptive_smdegrain",
)
core = vs.core

DEGRAIN_KERNEL_KWARGS = {
    "smdegrain": [
        "tr",
        "thSAD",
        "thSADC",
        "RefineMotion",
        "contrasharp",
        "CClip",
        "interlaced",
        "tff",
        "plane",
        "Globals",
        "pel",
        "subpixel",
        "prefilter",
        "mfilter",
        "blksize",
        "overlap",
        "search",
        "truemotion",
        "MVglobal",
        "dct",
        "limit",
        "limitc",
        "thSCD1",
        "thSCD2",
        "chroma",
        "hpad",
        "vpad",
        "Str",
        "Amp",
    ],
    "knlmeanscl": [
        "d",
        "a",
        "s",
        "h",
        "channels",
        "wmode",
        "wref",
        "rclip",
        "device_type",
        "device_id",
        "ocl_x",
        "ocl_y",
        "ocl_r",
        "info",
    ],
    "tnlmeanscl": ["ax", "ay", "az", "sx", "sy", "bx", "by", "a", "h", "ssd"],
    "bm3d": [
        "sigma",
        "radius1",
        "radius2",
        "profile1",
        "profile2",
        "refine",
        "pre",
        "ref",
        "psample",
        "matrix",
        "full",
        "output",
        "css",
        "depth",
        "sample",
        "dither",
        "useZ",
        "prefer_props",
        "ampo",
        "ampn",
        "dyn",
        "staticnoise",
        "cu_kernel",
        "cu_taps",
        "cu_a1",
        "cu_a2",
        "cu_cplace",
        "cd_kernel",
        "cd_taps",
        "cd_a1",
        "cd_a2",
        "cd_cplace",
        "cd_a1",
        "cd_a2",
        "cd_cplace",
        "block_size1",
        "block_step1",
        "group_size1",
        "bm_range1",
        "bm_step1",
        "ps_num1",
        "ps_range1",
        "ps_step1",
        "th_mse1",
        "block_size2",
        "block_step2",
        "group_size2",
        "bm_range2",
        "bm_step2",
        "ps_num2",
        "ps_range2",
        "ps_step2",
        "th_mse2",
        "hard_thr",
    ],
    "dfttest": [
        "ftype",
        "sigma",
        "sigma2",
        "pmin",
        "pmax",
        "sbsize",
        "smode",
        "sosize",
        "tbsize",
        "tmode",
        "tosize",
        "swin",
        "twin",
        "sbeta",
        "tbeta",
        "zmean",
        "f0beta",
        "nstring",
        "sstring",
        "ssx",
        "ssy",
        "sst",
        "planes",
        "opt",
    ],
}

VALID_DEGRAIN_KERNELS = {
    "smd": "smdegrain",
    "bm3d": "bm3d",
    "dft": "dfttest",
    "knlm": "knlmeanscl",
    "tnlm": "tnlmeanscl",
    "knl": "knlmeanscl",
    "tnl": "tnlmeanscl",
    "smdegrain": "smdegrain",
    "knlmeanscl": "knlmeanscl",
    "tnlmeanscl": "tnlmeanscl",
    "dfttest": "dfttest",
}
DegrainKernal = Literal[
    "smdegrain",
    "smd",
    "bm3d",
    "knlmeanscl",
    "knlm",
    "knl",
    "tnlmeanscl",
    "tnlm",
    "tnl",
    "dfttest",
    "dft",
]
DegrainArea = Literal["light", "dark"]


def adaptive_degrain2(
    src: vs.VideoNode,
    luma_scaling: int = 30,
    kernel: DegrainKernal = "smdegrain",
    area: DegrainArea = "light",
    iter_edge: int = 0,
    show_mask: bool = False,
    **degrain_kwargs: Dict[str, Any],
) -> vs.VideoNode:
    """
    An adaptive degrainer that took kageru adaptive_grain function
    and apply degrainer that you picked.

    Available degrain kernel are:
    - SMDegrain (from havsfunc)
    - KNLMeansCL
    - TNLMeansCL
    - BM3D (from mvsfunc)
    - DFTTest

    Parameters
    ----------
    src: :class:`VideoNode`
        The video to be degrained.
    luma_scaling: :class:`int`
        The mask luma scaling that will be used.
    kernel: :class:`DegrainKernal`
        The degrainer kernel that you want to use.
    area: :class:`DegrainArea`
        The area that you want to use.
    iter_edge: :class:`int`
        How many time we will need to iterate the edge mask.
    show_mask: :class:`bool`
        Do you want to show the mask or not.
    **degrain_kwargs: :class:`Dict[str, Any]`
        The arguments that will be passed to the degrainer.

    Returns
    -------
    :class:`VideoNode`
        The degrained video.
    """

    if not isinstance(src, vs.VideoNode):
        raise TypeError("adaptive_degrain2: src must be a clip")
    area = area.lower()  # type: ignore
    if area not in ("light", "dark"):
        raise ValueError("adaptive_degrain2: area must be `light` or `dark`")

    kernel = kernel.lower()  # type: ignore
    if kernel not in VALID_DEGRAIN_KERNELS:
        raise ValueError("adaptive_degrain2: kernel must be one of {}".format(VALID_DEGRAIN_KERNELS))

    kernel = VALID_DEGRAIN_KERNELS[kernel]  # type: ignore

    for argument in degrain_kwargs:
        if argument not in DEGRAIN_KERNEL_KWARGS[kernel]:
            raise ValueError("adaptive_degrain2: {} is not a valid argument for {}".format(argument, kernel))

    degrain_func: Callable[[vs.VideoNode], vs.VideoNode]
    if kernel == "smdegrain":
        SMDegrain = try_import("havsfunc", "SMDegrain")
        if SMDegrain is None:
            raise ImportError(
                "adaptive_degrain2: SMDegrain is not available and the selected kernel is SMDegrain"
            )
        degrain_func = lambda src: SMDegrain(src, **degrain_kwargs)  # type: ignore
    elif kernel == "knlmeanscl":
        degrain_func = lambda src: core.knlm.KNLMeansCL(src, **degrain_kwargs)
    elif kernel == "tnlmeanscl":
        degrain_func = lambda src: core.tnlm.TNLMeans(src, **degrain_kwargs)
    elif kernel == "dfttest":
        degrain_func = lambda src: core.dfttest.DFTTest(src, **degrain_kwargs)
    elif kernel == "bm3d":
        BM3D = try_import("mvsfunc", "BM3D")
        if BM3D is None:
            raise ImportError("adaptive_degrain2: BM3D is not available and the selected kernel is BM3D")
        degrain_func = lambda src: BM3D(src, **degrain_kwargs)  # type: ignore
    else:
        raise ValueError("adaptive_degrain2: kernel must be one of {}".format(VALID_DEGRAIN_KERNELS))

    adapt_mask = core.adg.Mask(src.std.PlaneStats(), luma_scaling)
    y_plane = get_y(src)

    if area == "light":
        adapt_mask = adapt_mask.std.Invert()

    # fmt: off
    limitx = y_plane.std.Convolution(
        [
            -1, -2, -1,  # noqa: E241,E131
             0,  0,  0,  # noqa: E241,E131
             1,  2,  1,  # noqa: E241,E131
        ],
        saturate=False,
    )
    limity = y_plane.std.Convolution(
        [
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1,
        ],
        saturate=False,
    )
    # fmt: on
    limit = core.std.Expr([limitx, limity], "x y max")
    limit = iterate(limit, core.std.Inflate, iter_edge)

    mask = core.std.Expr([adapt_mask, limit], "x y -")

    if show_mask:
        return mask

    fil = degrain_func(src)
    return core.std.MaskedMerge(src, fil, mask)


adaptive_bm3d = partial(adaptive_degrain2, kernel="bm3d")
adaptive_dfttest = partial(adaptive_degrain2, kernel="dfttest")
adaptive_knlm = partial(adaptive_degrain2, kernel="knlm")
adaptive_tnlm = partial(adaptive_degrain2, kernel="tnlm")
adaptive_smdegrain = partial(adaptive_degrain2, kernel="smd")
