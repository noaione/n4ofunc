from os import path as os_path
from pathlib import Path
from typing import Union

import fvsfunc as fvf
import vapoursynth as vs
from vsutil import get_y, insert_clip, iterate

__all__ = (
    "antiedgemask",
    "simple_native_mask",
    "recursive_apply_mask",
    "rapplym",
    "antiedge",
    "native_mask",
)
core = vs.core
IntegerFloat = Union[int, float]


def antiedgemask(src: vs.VideoNode, iteration: int = 1) -> vs.VideoNode:
    """
    Create an anti-edge mask that use a blank clip
    and Sobel to create a clip that does not include edges.

    Parameters
    ----------
    src: :class:`VideoNode`
        The video to be anti-edge masked.
    iteration: :class:`int`
        How many times we will need to iterate the anti-edge mask.

    Returns
    -------
    :class:`VideoNode`
        The anti-edge masked video.
    """
    if not isinstance(src, vs.VideoNode):
        raise TypeError("antiedgemask: src must be a clip")
    if not isinstance(iteration, int):
        raise TypeError("antiedgemask: iteration must be an integer")
    if iteration < 1:
        raise ValueError("antiedgemask: iteration must be greater than 0")

    w, h = src.width, src.height
    y_plane = get_y(src)

    white_clip = core.std.BlankClip(
        src, width=w, height=h, color=[255] * src.format.num_planes
    ).std.ShufflePlanes(0, vs.GRAY)
    edge_mask = core.std.Sobel(y_plane)
    edge_mask = iterate(edge_mask, core.std.Maximum, iteration)

    return core.std.Expr([white_clip, edge_mask], "x y -")


def simple_native_mask(
    clip: vs.VideoNode,
    descale_w: IntegerFloat,
    descale_h: IntegerFloat,
    blurh: IntegerFloat = 1.5,
    blurv: IntegerFloat = 1.5,
    iter_max: int = 3,
    no_resize: bool = False,
) -> vs.VideoNode:
    """
    Create a native mask to make sure native content does not get descaled.

    Parameters
    ----------
    clip: :class:`VideoNode`
        The video source.
    descale_w: :class:`Union[int, float]`
        Target descale width resolution for checking.
    descale_h: :class:`Union[int, float]`
        Target descale height resolution for checking.
    blurh: :class:`Union[int, float]`
        Horizontal blur strength.
    blurv: :class:`Union[int, float]`
        Vertical blur strength.
    iter_max: :class:`int`
        Iteration count that will expand the mask size.
    no_resize: :class:`bool`
        Don't resize to the descaled resolution (keep it at original resolution)

    Returns
    -------
    :class:`VideoNode`
        The native mask.
    """
    clip_32 = fvf.Depth(clip, 32)
    y_32 = get_y(clip_32)
    clip_bits = clip.format.bits_per_sample

    target_w = clip.width
    target_h = clip.height

    down = core.descale.Debicubic(y_32, descale_w, descale_h)
    up = core.resize.Bicubic(down, target_w, target_h)
    dmask = core.std.Expr([y_32, up], "x y - abs 0.025 > 1 0 ?")
    dmask = iterate(dmask, core.std.Maximum, iter_max)
    if blurh > 0 and blurv > 0:
        dmask = core.std.BoxBlur(dmask, hradius=blurh, vradius=blurv)
    if not no_resize:
        dmask = core.resize.Bicubic(dmask, descale_w, descale_h)
    return fvf.Depth(dmask, clip_bits)


def recursive_apply_mask(
    src_a: vs.VideoNode,
    src_b: vs.VideoNode,
    mask_folder: Union[str, Path],
    iter: int = 1,
) -> vs.VideoNode:
    """
    Recursively check `mask_folder` for a .png or .ass file
    After it found all of them, it will use the mask
    to merge together the two clips.

    Acceptable filename format:
    - frameNum.png
    - frameStart-frameEnd.png
    - itsUpToYou.ass
    Example:
    - 2500.png
    - 2000-2004.png
    - maskep1.ass

    Parameters
    ----------
    src_a: :class:`VideoNode`
        The first clip.
    src_b: :class:`VideoNode`
        The second clip.
    mask_folder: :class:`Union[str, Path]`
        The folder that contains the masks.
    iter: :class:`int`
        How many times we will need to iterate the mask.

    Returns
    -------
    :class:`VideoNode`
        The merged clip.
    """

    if isinstance(mask_folder, str):
        mask_folder = Path(mask_folder)

    imwri = core.imwri
    masks_png = mask_folder.glob("*.png")
    for mask in masks_png:
        frame = os_path.basename(mask).rsplit(".", 1)[0]
        frame = [int(i) for i in frame.split("-")][:2]
        if len(frame) < 2:
            frame = [frame[0], frame[0]]
        first_f, last_f = frame

        image = fvf.Depth(
            imwri.Read(mask).resize.Point(
                format=vs.GRAYS, matrix_s="709"
            ),
            src_a.format.bits_per_sample
        ).std.AssumeFPS(
            fpsnum=src_a.fps.numerator,
            fpsden=src_a.fps.denominator,
        )

        src_a_n, src_b_n = src_a[first_f:last_f + 1], src_b[first_f:last_f + 1]

        image = image * ((last_f + 1) - first_f)
        image = get_y(image)
        src_masked = core.std.MaskedMerge(src_a_n, src_b_n, image)
        for _ in range(iter - 1):
            src_masked = src_masked.std.MaskedMerge(src_masked, src_b_n, image)
        src_a = insert_clip(src_a, insert=src_masked, start_frame=first_f)

    masks_ass = mask_folder.glob("*.ass")
    for mask in masks_ass:
        blank_mask = src_a.std.BlankClip()
        ass_mask = get_y(blank_mask.sub.TextFile(mask))

        src_a = core.std.MaskedMerge(src_a, src_b, ass_mask)
    return src_a


rapplym = recursive_apply_mask
antiedge = antiedgemask
native_mask = simple_native_mask
