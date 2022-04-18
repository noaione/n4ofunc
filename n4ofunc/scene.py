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

import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import vapoursynth as vs
from fvsfunc import Depth
from vsutil import plane

from .utils import is_extension

__all__ = ("scene_filter",)
core = vs.core


def scene_filter(
    clip: vs.VideoNode,
    mappings: str,
    filter_fn: Callable[..., vs.VideoNode],
    filter_mask: Optional[str] = None,
    *filter_args: List[Any],
    **filter_kwargs: Dict[str, Any],
) -> vs.VideoNode:
    """
    Apply a specific function to a specific frame of a clip.

    Parameters
    ----------
    clip: :class:`VideoNode`
        The source or input clip.
    mappings: :class:`str`
        The frame mappings that will be used, refer to previous section
        for how to enter the mappings format.
    filter_fn: :class:`Callable[..., VideoNode]`
        The function that will be applied to the clip.
    filter_mask: :class:`Optional[str]`
        The mask that will be used when applying the filter (if needed).
    filter_args: :class:`List[Any]`
        The arguments that will be passed to the filter function.
    filter_kwargs: :class:`Dict[str, Any]`
        The keyword arguments that will be passed to the filter function.

    Returns
    -------
    :class:`VideoNode`
        The output clip.

    """
    if not isinstance(clip, vs.VideoNode):
        raise ValueError("scene_filter: 'clip' must be a clip.")

    cbits = clip.format.bits_per_sample
    cframes = clip.num_frames

    if cbits < 16:
        clip = Depth(clip, 16)

    mappings = mappings.replace(",", " ").replace(":", " ")
    frame_maps = re.findall(r"\d+(?!\d*\s*\d*\s*\d*\])", mappings)
    range_maps = re.findall(r"\[\s*\d+\s+\d+\s*\]", mappings)

    actual_maps: List[int] = []
    for frame_map in frame_maps:
        actual_maps.append(int(frame_map))
    for range_map in range_maps:
        start, end = range_map.strip("[]").split(" ")
        actual_maps.extend(list(range(int(start), int(end) + 1)))

    for frame in actual_maps:
        if frame >= cframes:
            raise ValueError(f"scene_filter: 'mappings' contains invalid frame {frame}.")

    mask_clip: Optional[vs.VideoNode] = None
    skip_frame_eval = False
    if filter_mask:
        if is_extension(filter_mask, "ass"):
            skip_frame_eval = True
            b_clip = core.std.BlankClip(width=clip.width, height=clip.height, length=cframes)
            mask_clip = core.sub.TextFile(b_clip, filter_mask).std.Binarize()
        else:
            mask_clip = core.imwri.Read(filter_mask)
        mask_clip = plane(mask_clip, 0)
        mask_clip = Depth(mask_clip, 16)
        mask_clip = core.std.BoxBlur(mask_clip, hradius=5, vradius=5)

    if skip_frame_eval and mask_clip is not None:
        ref = clip
        fil = filter_fn(ref, *filter_args, **filter_kwargs)
        return core.std.MaskedMerge(ref, fil, mask_clip)

    def scene_filter_func(n: int, c: vs.VideoNode):
        if n not in actual_maps:
            return c[n]
        ref = c[n]
        fil = filter_fn(ref, *filter_args, **filter_kwargs)
        if mask_clip is not None:
            fil = core.std.MaskedMerge(ref, fil, mask_clip)
        return fil

    return core.std.FrameEval(clip, partial(scene_filter_func, c=clip))
