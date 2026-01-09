'''
Module Name : curlyBrace

Author : 高斯羽 博士 (Dr. GAO, Siyu)

Version : 1.0.2

Last Modified : 2019-04-22

This module is basically a Python implementation of the function written Pål Næverlid Sævik
for MATLAB (link in Reference).

The function "curlyBrace" allows you to plot an optionally annotated curly bracket between
two points when using matplotlib.

The usual settings for line and fonts in matplotlib also apply.

The function takes the axes scales into account automatically. But when the axes aspect is
set to "equal", the auto switch should be turned off.

Change Log
----------------------
* **Notable changes:**
    + Version : 1.0.2
        - Added considerations for different scaled axes and log scale
    + Version : 1.0.1
        - First version.

Reference
----------------------
https://uk.mathworks.com/matlabcentral/fileexchange/38716-curly-brace-annotation

List of functions
----------------------

* getAxSize_
* curlyBrace_

'''

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


FloatArray = npt.NDArray[np.floating[Any]]
ScalarFunc = Callable[[float], float]
ArrayFunc = Callable[[FloatArray], FloatArray]


@overload
def mirroring(vals: float, function: ScalarFunc) -> float: ...


@overload
def mirroring(vals: FloatArray, function: ArrayFunc) -> FloatArray: ...


def mirroring(
    vals: float | FloatArray,
    function: ScalarFunc | ArrayFunc,
) -> float | FloatArray:
    """Apply a function with sign preservation for negative values.

    For positive values: returns function(val)
    For negative values: returns -function(abs(val))
    For zero: returns 0.0

    This allows applying log/exp transforms to arrays that may contain negative values
    by mirroring the transformation across zero.
    """
    if isinstance(vals, (float, np.floating)):
        scalar_func = cast('ScalarFunc', function)
        scalar_val = float(vals)
        if scalar_val > 0.0:
            return scalar_func(scalar_val)
        if scalar_val < 0.0:
            return -scalar_func(abs(scalar_val))
        return 0.0

    array_vals = cast('FloatArray', vals)
    array_func = cast('ArrayFunc', function)

    result = np.zeros_like(array_vals, dtype=float)

    pos_mask = array_vals > 0.0
    neg_mask = array_vals < 0.0

    if np.any(pos_mask):
        result[pos_mask] = array_func(array_vals[pos_mask])
    if np.any(neg_mask):
        result[neg_mask] = -array_func(np.abs(array_vals[neg_mask]))
    return result


def add_bracket_annotation(
    ax: matplotlib.axes.Axes,
    str_text: str,
    summit_x: float,
    summit_y: float,
    theta: float,
    ax_ylim: list[float],
    int_line_num: int = 2,
    fontdict: dict[str, Any] | None = None,
) -> None:
    """Add text annotation to the bracket at the summit position.

    Parameters
    ----------
    ax : matplotlib axes object
        The target axes.
    str_text : str
        The annotation text to display.
    summit_x : float
        X-coordinate of the bracket summit (annotation position).
    summit_y : float
        Y-coordinate of the bracket summit (annotation position).
    theta : float
        The bracket angle in radians.
    ax_ylim : list[float]
        The y-axis limits to determine axis orientation.
    int_line_num : int
        Number of lines spacing between bracket and text.
    fontdict : dict[str, Any] | None
        Font dictionary for text styling.
    """
    if fontdict is None:
        fontdict = {}

    int_line_num = int(int_line_num)
    str_temp = '\n' * int_line_num

    # Convert radians to degree and within 0 to 360
    ang = np.degrees(theta) % 360.0

    # Determine rotation and text position based on angle and axis orientation
    if 0.0 <= ang <= 90.0:
        if ax_ylim[0] < ax_ylim[1]:
            rotation = ang
            str_text = str_text + str_temp
        else:
            rotation = -ang
            str_text = str_temp + str_text
    elif 90.0 < ang < 270.0:
        if ax_ylim[0] < ax_ylim[1]:
            rotation = ang + 180.0
            str_text = str_temp + str_text
        else:
            rotation = -(ang + 180.0)
            str_text = str_text + str_temp
    elif 270.0 <= ang <= 360.0:
        if ax_ylim[0] < ax_ylim[1]:
            rotation = ang
            str_text = str_text + str_temp
        else:
            rotation = -ang
            str_text = str_temp + str_text
    else:
        rotation = ang if ax_ylim[0] < ax_ylim[1] else -ang

    ax.text(summit_x, summit_y, str_text, ha='center', va='center', rotation=rotation, fontdict=fontdict)


def getAxSize(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes) -> tuple[float, float]:
    '''
    .. _getAxSize :

    Get the axes size in pixels.

    Parameters
    ----------
    fig : matplotlib figure object
        The of the target axes.

    ax : matplotlib axes object
        The target axes.

    Returns
    -------
    ax_width : float
        The axes width in pixels.

    ax_height : float
        The axes height in pixels.

    Reference
    -----------
    https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    '''

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width, ax_height = bbox.width, bbox.height
    ax_width *= fig.dpi
    ax_height *= fig.dpi

    return ax_width, ax_height


class CurlyBraceResult(NamedTuple):
    theta: float
    summit: list[float]
    arc1: list[float]
    arc2: list[float]
    arc3: list[float]
    arc4: list[float]


def curlyBrace(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    p1: list[float] | tuple[float, float],
    p2: list[float] | tuple[float, float],
    k_r: float = 0.1,
    bool_auto: bool = True,
    str_text: str = '',
    int_line_num: int = 2,
    fontdict: dict[str, Any] | None = None,
    **kwargs,  # noqa: ANN003
) -> CurlyBraceResult:
    # def curlyBrace(fig, ax, p1, p2, k_r=0.1, bool_auto=True, str_text='', int_line_num=2, fontdict={}, **kwargs):
    '''
    .. _curlyBrace :

    Plot an optionally annotated curly bracket on the given axes of the given figure.

    Note that the brackets are anti-clockwise by default. To reverse the text position, swap
    "p1" and "p2".

    Note that, when the axes aspect is not set to "equal", the axes coordinates need to be
    transformed to screen coordinates, otherwise the arcs may not be seeable.

    Parameters
    ----------
    fig : matplotlib figure object
        The of the target axes.

    ax : matplotlib axes object
        The target axes.

    p1 : two element numeric list
        The coordinates of the starting point.

    p2 : two element numeric list
        The coordinates of the end point.

    k_r : float
        This is the gain controlling how "curvy" and "pointy" (height) the bracket is.

        Note that, if this gain is too big, the bracket would be very strange.

    bool_auto : boolean
        This is a switch controlling wether to use the auto calculation of axes
        scales.

        When the two axes do not have the same aspects, i.e., not "equal" scales,
        this should be turned on, i.e., True.

        When "equal" aspect is used, this should be turned off, i.e., False.

        If you do not set this to False when setting the axes aspect to "equal",
        the bracket will be in funny shape.

        Default = True

    str_text : string
        The annotation text of the bracket. It would displayed at the mid point
        of bracket with the same rotation as the bracket.

        By default, it follows the anti-clockwise convention. To flip it, swap
        the end point and the starting point.

        The appearance of this string can be set by using "fontdict", which follows
        the same syntax as the normal matplotlib syntax for font dictionary.

        Default = empty string (no annotation)

    int_line_num : int
        This argument determines how many lines the string annotation is from the summit
        of the bracket.

        The distance would be affected by the font size, since it basically just a number of
        lines appended to the given string.

        Default = 2

    fontdict : dictionary
        This is font dictionary setting the string annotation. It is the same as normal
        matplotlib font dictionary.

        Default = empty dict

    **kwargs : matplotlib line setting arguments
        This allows the user to set the line arguments using named arguments that are
        the same as in matplotlib.

    Returns
    -------
    theta : float
        The bracket angle in radians.

    summit : list
        The positions of the bracket summit.

    arc1 : list of lists
        arc1 positions.

    arc2 : list of lists
        arc2 positions.

    arc3 : list of lists
        arc3 positions.

    arc4 : list of lists
        arc4 positions.

    Reference
    ----------
    https://uk.mathworks.com/matlabcentral/fileexchange/38716-curly-brace-annotation
    '''
    if fontdict is None:
        fontdict = {}

    pt1: list[float | None] = [None, None]
    pt2: list[float | None] = [None, None]

    ax_width, ax_height = getAxSize(fig, ax)

    ax_xlim = list(ax.get_xlim())
    ax_ylim = list(ax.get_ylim())

    # log scale consideration
    if 'log' in ax.get_xaxis().get_scale():
        pt1[0] = mirroring(p1[0], np.log)
        pt2[0] = mirroring(p2[0], np.log)
        ax_xlim = mirroring(np.array(ax_xlim), np.log).tolist()
    else:
        pt1[0] = p1[0]
        pt2[0] = p2[0]

    if 'log' in ax.get_yaxis().get_scale():
        pt1[1] = mirroring(p1[1], np.log)
        pt2[1] = mirroring(p2[1], np.log)
        ax_ylim = mirroring(np.array(ax_ylim), np.log).tolist()
    else:
        pt1[1] = p1[1]
        pt2[1] = p2[1]

    # get the ratio of pixels/length
    xscale = ax_width / abs(ax_xlim[1] - ax_xlim[0])
    yscale = ax_height / abs(ax_ylim[1] - ax_ylim[0])

    # this is to deal with 'equal' axes aspects
    if not bool_auto:
        xscale = 1.0
        yscale = 1.0

    # convert length to pixels,
    # need to minus the lower limit to move the points back to the origin. Then add the limits back on end.
    pt1[0] = (pt1[0] - ax_xlim[0]) * xscale
    pt1[1] = (pt1[1] - ax_ylim[0]) * yscale
    pt2[0] = (pt2[0] - ax_xlim[0]) * xscale
    pt2[1] = (pt2[1] - ax_ylim[0]) * yscale

    # calculate the angle
    theta = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

    # calculate the radius of the arcs
    r = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) * k_r

    # arc1 centre
    x11 = pt1[0] + r * np.cos(theta)
    y11 = pt1[1] + r * np.sin(theta)

    # arc2 centre
    x22 = (pt2[0] + pt1[0]) * 0.5 - 2.0 * r * np.sin(theta) - r * np.cos(theta)
    y22 = (pt2[1] + pt1[1]) * 0.5 + 2.0 * r * np.cos(theta) - r * np.sin(theta)

    # arc3 centre
    x33 = (pt2[0] + pt1[0]) * 0.5 - 2.0 * r * np.sin(theta) + r * np.cos(theta)
    y33 = (pt2[1] + pt1[1]) * 0.5 + 2.0 * r * np.cos(theta) + r * np.sin(theta)

    # arc4 centre
    x44 = pt2[0] - r * np.cos(theta)
    y44 = pt2[1] - r * np.sin(theta)

    # prepare the rotated
    q = np.linspace(theta, theta + 0.5 * np.pi, 50)

    # reverse q
    t = np.flip(q)

    # arc coordinates
    arc1x = r * np.cos(t + 0.5 * np.pi) + x11
    arc1y = r * np.sin(t + 0.5 * np.pi) + y11

    arc2x = r * np.cos(q - 0.5 * np.pi) + x22
    arc2y = r * np.sin(q - 0.5 * np.pi) + y22

    arc3x = r * np.cos(q + np.pi) + x33
    arc3y = r * np.sin(q + np.pi) + y33

    arc4x = r * np.cos(t) + x44
    arc4y = r * np.sin(t) + y44

    # convert back to the axis coordinates
    arc1x = arc1x / xscale + ax_xlim[0]
    arc2x = arc2x / xscale + ax_xlim[0]
    arc3x = arc3x / xscale + ax_xlim[0]
    arc4x = arc4x / xscale + ax_xlim[0]

    arc1y = arc1y / yscale + ax_ylim[0]
    arc2y = arc2y / yscale + ax_ylim[0]
    arc3y = arc3y / yscale + ax_ylim[0]
    arc4y = arc4y / yscale + ax_ylim[0]

    # log scale consideration - convert back from log space
    if 'log' in ax.get_xaxis().get_scale():
        arc1x = mirroring(arc1x, np.exp)
        arc2x = mirroring(arc2x, np.exp)
        arc3x = mirroring(arc3x, np.exp)
        arc4x = mirroring(arc4x, np.exp)

    if 'log' in ax.get_yaxis().get_scale():
        arc1y = mirroring(arc1y, np.exp)
        arc2y = mirroring(arc2y, np.exp)
        arc3y = mirroring(arc3y, np.exp)
        arc4y = mirroring(arc4y, np.exp)

    # plot arcs - plot first arc and extract its color to ensure all parts match
    (first_arc_line,) = ax.plot(arc1x, arc1y, **kwargs)
    bracket_color = first_arc_line.get_color()
    kwargs["color"] = bracket_color

    ax.plot(arc2x, arc2y, **kwargs)
    ax.plot(arc3x, arc3y, **kwargs)
    ax.plot(arc4x, arc4y, **kwargs)

    # plot connecting lines between arcs
    ax.plot([arc1x[-1], arc2x[1]], [arc1y[-1], arc2y[1]], **kwargs)
    ax.plot([arc3x[-1], arc4x[1]], [arc3y[-1], arc4y[1]], **kwargs)

    summit = [arc2x[-1], arc2y[-1]]

    if str_text:
        add_bracket_annotation(ax, str_text, arc2x[-1], arc2y[-1], theta, ax_ylim, int_line_num, fontdict)

    arc1 = [arc1x, arc1y]
    arc2 = [arc2x, arc2y]
    arc3 = [arc3x, arc3y]
    arc4 = [arc4x, arc4y]

    return CurlyBraceResult(theta, summit, arc1, arc2, arc3, arc4)
