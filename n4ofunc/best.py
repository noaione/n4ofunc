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
from typing import Dict, List, Literal, Optional, Union

import vapoursynth as vs

__all__ = (
    "better_planes",
    "better_frames",
    "better_frame",
    "bplanes",
    "bframes",
    "bframe",
)
core = vs.core
PropType = Literal[
    "average",
    "maximum",
    "minimum",
    "addition",
    "subtract",
    "avg",
    "max",
    "min",
    "add",
    "sub",
]
PropTypeActual = Literal[
    "PlaneStatsMax",
    "PlaneStatsMin",
    "PlaneStatsAverage",
    "BothAdd",
    "BothSubtract",
]


def _source_frame_selector(
    n: int, f: List[vs.VideoFrame], clips: List[vs.VideoNode], prop_type: PropTypeActual
) -> vs.VideoNode:
    clip_data: List[float] = []
    for p in f:
        if prop_type == "BothAdd":
            clip_data.append(p.props["PlaneStatsMax"] + p.props["PlaneStatsMin"])
        elif prop_type == "BothSubtract":
            clip_data.append(p.props["PlaneStatsMax"] - p.props["PlaneStatsMin"])
        else:
            clip_data.append(p.props[prop_type])
    return clips[clip_data.index(max(clip_data))]


def better_planes(
    clips: List[vs.VideoNode],
    props: Union[PropType, List[PropType]] = "avg",
    show_info: Optional[Union[bool, List[str]]] = None,
) -> vs.VideoNode:
    """
    A naive function to pick the best frames from a list of sources.

    This only check luma plane (Y plane) not like better_planes that check every plane
    Then using defined ``props`` they will be compared:
    - Avg: Highest Average PlaneStats
    - Min: Lowest Minimum PlaneStats
    - Max: Highest Maximum PlaneStats
    - Add: Value from combining PlaneStatsMax with PlaneStatsMin
    - Sub: Value from subtracting PlaneStatsMax with PlaneStatsMin

    The best outcome plane will be returned.

    ``props`` value must be:
    - For using PlaneStatsAverage as comparasion: "avg", "average", or "planestatsaverage"
    - For using PlaneStatsMin as comparasion: "min", "minimum", or "planestatsmin"
    - For using PlaneStatsMax as comparasion: "max", "maximum", or "planestatsmax"
    - For subtracting PlaneStatsMax with PlaneStatsMin as comparasion: "sub" or "subtract"
    - For combining value of PlaneStatsMax with PlaneStatsMin as comparasion: "add" or "addition"

    Examples: ::

        src = nao.better_planes(
            clips=[src_vbr, src_hevc, src_cbr], props=["max", "avg"],
            show_info=["AMZN VBR", "AMZN HEVC", "AMZN CBR"]
        )
        # Or
        src = nao.better_planes(clips=[src1, src2, src3], show_info=["CR", "Funi", "Abema"])
        # Or
        src = nao.better_planes(clips=[src1, src2, src3], show_info=True)
        # Or
        src = nao.better_planes(clips=[src1, src2], props="max")

    Parameters
    ----------
    clips: :class:`List[VideoNode]`
        List of source to be used.
    props: :class:`Union[PropType, List[PropType]]`
        A list or a string of what will be used for comparing.
        If it's a list, the maximum is 3 (For Y, U, and V)
    show_info: :class:`Optional[Union[bool, List[str]]]`
        Show text for what sources are used, if it's boolean it will
        use a numbering system.

    Returns
    -------
    :class:`VideoNode`
        The combined clip from multiple sources.
    """
    if not isinstance(clips, list):
        raise TypeError("better_planes: clips must be a list of sources")
    if not isinstance(props, (list, str)):
        raise TypeError("better_planes: props must be a list or a string")
    if show_info is not None and not isinstance(show_info, (bool, list)):
        raise TypeError("better_planes: show_info must be a boolean or list of string")
    if isinstance(show_info, list) and len(show_info) != len(clips):
        raise ValueError(
            "better_planes: show_info must be a boolean or list of string" "with the same length as clips"
        )

    allowed_props: Dict[PropType, PropTypeActual] = {
        "avg": "PlaneStatsAverage",
        "min": "PlaneStatsMin",
        "max": "PlaneStatsMax",
        "average": "PlaneStatsAverage",
        "minimum": "PlaneStatsMin",
        "maximum": "PlaneStatsMax",
        "planestatsaverage": "PlaneStatsAverage",
        "planestatsmin": "PlaneStatsMin",
        "planestatsmax": "PlaneStatsMax",
        "add": "BothAdd",
        "sub": "BothSubtract",
        "addition": "BothAdd",
        "subtract": "BothSubtract",
    }

    if isinstance(props, str):
        props = props.lower()
        if props not in allowed_props:
            raise ValueError(
                "better_planes: `props` must be a {}".format("`" + "` or `".join(list(allowed_props.keys())))
                + "`"
            )
        props_ = [allowed_props[props] for _ in range(3)]
    else:
        up_t = len(props)
        props_ = []
        if up_t == 1:
            t_props_ = [props[0] for i in range(3)]
        elif up_t == 2:
            t_props_ = [props[0], props[1], props[1]]
        elif up_t == 3:
            t_props_ = props
        elif up_t > 3:
            t_props_ = props[:3]

        for n, i in enumerate(t_props_):
            if i in allowed_props:
                props_.append(allowed_props[i.lower()])
            else:
                raise ValueError(
                    "better_planes: `props[{}]` must be a {}".format(
                        n, "`" + "` or `".join(list(allowed_props.keys()))
                    )
                    + "`"
                )

    y_clips: List[vs.VideoNode] = []
    u_clips: List[vs.VideoNode] = []
    v_clips: List[vs.VideoNode] = []

    if isinstance(show_info, list):
        for n, clip in enumerate(clips):
            y_clips.append(
                core.std.Text(
                    core.std.ShufflePlanes(clip, 0, vs.GRAY),
                    f"{show_info[n]} - Y {props_[0]}",
                    7,
                )
            )
            u_clips.append(
                core.std.Text(
                    core.std.ShufflePlanes(clip, 1, vs.GRAY),
                    f"{show_info[n]} - Y {props_[1]}",
                    8,
                )
            )
            v_clips.append(
                core.std.Text(
                    core.std.ShufflePlanes(clip, 2, vs.GRAY),
                    f"{show_info[n]} - Y {props_[2]}",
                    9,
                )
            )
    elif show_info:
        for n, clip in enumerate(clips, 1):
            y_clips.append(
                core.text.Text(
                    core.std.ShufflePlanes(clip, 0, vs.GRAY),
                    f"Input {n} - Y {props_[0]}",
                    7,
                )
            )
            u_clips.append(
                core.text.Text(
                    core.std.ShufflePlanes(clip, 1, vs.GRAY),
                    f"Input {n} - U {props_[1]}",
                    8,
                )
            )
            v_clips.append(
                core.std.Text(
                    core.std.ShufflePlanes(clip, 2, vs.GRAY),
                    f"Input {n} - V {props_[2]}",
                    9,
                )
            )
    else:
        for clip in clips:
            y_clips.append(core.std.ShufflePlanes(clip, 0, vs.GRAY))
            u_clips.append(core.std.ShufflePlanes(clip, 1, vs.GRAY))
            v_clips.append(core.std.ShufflePlanes(clip, 2, vs.GRAY))

    y_clips_prop: List[vs.VideoNode] = []
    u_clips_prop: List[vs.VideoNode] = []
    v_clips_prop: List[vs.VideoNode] = []
    for clip in y_clips:
        y_clips_prop.append(clip.std.PlaneStats(plane=0))
    for clip in u_clips:
        u_clips_prop.append(clip.std.PlaneStats(plane=0))
    for clip in v_clips:
        v_clips_prop.append(clip.std.PlaneStats(plane=0))

    y_val = core.std.FrameEval(
        y_clips[0], partial(_source_frame_selector, clips=y_clips, prop_type=props_[0]), prop_src=y_clips_prop
    )
    u_val = core.std.FrameEval(
        u_clips[0], partial(_source_frame_selector, clips=u_clips, prop_type=props_[1]), prop_src=u_clips_prop
    )
    v_val = core.std.FrameEval(
        v_clips[0], partial(_source_frame_selector, clips=v_clips, prop_type=props_[2]), prop_src=v_clips_prop
    )
    return core.std.ShufflePlanes([y_val, u_val, v_val], [0, 1, 2], vs.YUV)


# TODO: Maybe add chroma support or something
def better_frames(
    clips: List[vs.VideoNode],
    props: Union[PropType, List[PropType]] = "avg",
    show_info: Optional[Union[bool, List[str]]] = None,
):
    """
    A naive function to pick the best planes out of every frame
    from a list of clip source.

    Every clips source planes are split into Y, U, and V.
    Then using defined ``props`` they will be compared:
    - Avg: Highest Average PlaneStats
    - Min: Lowest Minimum PlaneStats
    - Max: Highest Maximum PlaneStats
    - Add: Value from combining PlaneStatsMax with PlaneStatsMin
    - Sub: Value from subtracting PlaneStatsMax with PlaneStatsMin

    The best outcome plane will be returned.

    ``props`` value must be:
    - For using PlaneStatsAverage as comparasion: "avg", "average", or "planestatsaverage"
    - For using PlaneStatsMin as comparasion: "min", "minimum", or "planestatsmin"
    - For using PlaneStatsMax as comparasion: "max", "maximum", or "planestatsmax"
    - For subtracting PlaneStatsMax with PlaneStatsMin as comparasion: "sub" or "subtract"
    - For combining value of PlaneStatsMax with PlaneStatsMin as comparasion: "add" or "addition"

    Examples: ::

        src = nao.better_frame(
            clips=[src_vbr, src_hevc, src_cbr],
            props="add",
            show_info=["AMZN VBR", "AMZN HEVC", "AMZN CBR"]
        )
        # Or
        src = nao.better_frame(clips=[src1, src2, src3], show_info=["CR", "Funi", "Abema"])
        # Or
        src = nao.better_frame(clips=[src1, src2, src3], show_info=True)
        # Or
        src = nao.better_frame(clips=[src1, src2], props="max")

    Parameters
    ----------
    clips: :class:`List[VideoNode]`
        List of source to be used.
    props: :class:`PropType`
        A string of what will be used for comparing.
    show_info: :class:`Optional[Union[bool, List[str]]]`
        Show text for what sources are used, if it's boolean it will
        use a numbering system.

    Returns
    -------
    :class:`VideoNode`
        The combined clip from multiple sources.
    """
    if not isinstance(clips, list):
        raise TypeError("better_frames: clips must be a list of sources")
    if not isinstance(props, str):
        raise TypeError("better_frames: props must be a string")
    if show_info is not None and not isinstance(show_info, (bool, list)):
        raise TypeError("better_frames: show_info must be a boolean or list of string")
    if isinstance(show_info, list) and len(show_info) != len(clips):
        raise ValueError(
            "better_frames: show_info must be a boolean or list of string" "with the same length as clips"
        )

    allowed_props: Dict[PropType, PropTypeActual] = {
        "avg": "PlaneStatsAverage",
        "min": "PlaneStatsMin",
        "max": "PlaneStatsMax",
        "average": "PlaneStatsAverage",
        "minimum": "PlaneStatsMin",
        "maximum": "PlaneStatsMax",
        "planestatsaverage": "PlaneStatsAverage",
        "planestatsmin": "PlaneStatsMin",
        "planestatsmax": "PlaneStatsMax",
        "add": "BothAdd",
        "sub": "BothSubtract",
        "addition": "BothAdd",
        "subtract": "BothSubtract",
    }

    if isinstance(props, str):
        props = props.lower()
        if props not in allowed_props:
            raise ValueError(
                "better_frames: `props` must be a {}".format("`" + "` or `".join(list(allowed_props.keys())))
                + "`"
            )
        props_ = allowed_props[props]

    fmt_clips: List[vs.VideoNode] = []
    if isinstance(show_info, list):
        for n, clip in enumerate(clips):
            clips.append(
                core.text.Text(
                    clip,
                    f"{show_info[n]} - ({props_})",
                    7,
                )
            )
    elif show_info:
        for n, clip in enumerate(clips, 1):
            clips.append(
                core.text.Text(
                    clip,
                    f"Input {n} - ({props_})",
                    7,
                )
            )
    else:
        fmt_clips = clips

    clips_props: List[vs.VideoNode] = []
    for clip in fmt_clips:
        clips_props.append(clip.std.PlaneStats(plane=0))
    return core.std.FrameEval(
        fmt_clips[0], partial(_source_frame_selector, clips=fmt_clips, prop_type=props_), prop_src=clips_props
    )


bplanes = better_planes
better_frame = better_frames
bframe = better_frames
bframes = better_frames
