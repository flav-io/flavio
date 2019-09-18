"""Colour schemes for plots and colour utility functions."""

import matplotlib


# this was generated with the brewer2mpl package by calling
# brewer2mpl.get_map('Pastel1', 'qualitative', 9).mpl_colors
pastel = [(0.9843, 0.7059, 0.6824),
 (0.702, 0.8039, 0.8902),
 (0.8, 0.9216, 0.7725),
 (0.8706, 0.7961, 0.8941),
 (0.9961, 0.851, 0.651),
 (1, 1, 0.8),
 (0.898, 0.8471, 0.7412),
 (0.9922, 0.8549, 0.9255),
 (0.949, 0.949, 0.949)]

# this was generated with the brewer2mpl package by calling
# brewer2mpl.get_map('OrRd', 'sequential', 9).mpl_colors
reds = [(1.0, 0.9686274509803922, 0.9254901960784314),
 (0.996078431372549, 0.9098039215686274, 0.7843137254901961),
 (0.9921568627450981, 0.8313725490196079, 0.6196078431372549),
 (0.9921568627450981, 0.7333333333333333, 0.5176470588235295),
 (0.9882352941176471, 0.5529411764705883, 0.34901960784313724),
 (0.9372549019607843, 0.396078431372549, 0.2823529411764706),
 (0.8431372549019608, 0.18823529411764706, 0.12156862745098039),
 (0.7019607843137254, 0.0, 0.0),
 (0.4980392156862745, 0.0, 0.0)]

# this was generated with the brewer2mpl package by calling
# brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors
set1 = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
 (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
 (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (1.0, 0.4980392156862745, 0.0),
 (1.0, 1.0, 0.2),
 (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
 (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
 (0.6, 0.6, 0.6)]


# darken and lighten functions taken from https://github.com/BagelOrb/CompressionTestAnalysis/blob/master/PlottingUtil.py

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def darken_color(color, amount=0.5):
    """
    Darkens the given color by multiplying luminosity by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], amount * c[1], c[2])


def get_color(col, color):
    """Function needed for backwards compatibility with the old "col" argument in
    plt functions. It returns the default color 'C0' if both arguments are None.
    If 'color' is not None, it always uses that. If 'color' is None and 
    'col' is an integer, it returns the corresponding 'CN' color. If 'col' is
    neither None nor integer, an error is raised."""
    if color is None and col is None:
        return 'C0'
    if col is None:
        return color
    if not isinstance(col, int):
        raise ValueError("`col` must be an integer. Consider using `color` instead.")
    return 'C{}'.format(col)