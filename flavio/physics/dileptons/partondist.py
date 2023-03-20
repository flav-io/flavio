"""Functions for parton distributions."""


import parton
from functools import lru_cache
import scipy.interpolate
import numpy as np
from flavio.config import config

# temporary monkey patch to implement https://github.com/DavidMStraub/parton/pull/8
class MyRectBivariateSpline(scipy.interpolate.RectBivariateSpline):
    """Patch of the `scipy.interpolate.RectBivariateSpline` class extending
    it by the `bounds_error` and `fill_value` options that work
    like for `interp2d`."""

    def __init__(self, x, y, z, *args, bounds_error=False, fill_value=None,
                 **kwargs):
        """Initialize the `MyRectBivariateSpline instance`.
        The additional parameters `bounds_error` and `fill_value` work
        like for `interp2d`."""
        super().__init__(x, y, z, *args, **kwargs)
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.x_min, self.x_max = np.amin(x), np.amax(x)
        self.y_min, self.y_max = np.amin(y), np.amax(y)

        # a small margin is added to the min and max values to avoid numerical issues
        self.x_min = self.x_min - abs(self.x_min)/1e10
        self.x_max = self.x_max + abs(self.x_max)/1e10
        self.y_min = self.y_min - abs(self.y_min)/1e10
        self.y_max = self.y_max + abs(self.y_max)/1e10

    def __call__(self, x, y, *args, **kwargs):
        """Call the `MyRectBivariateSpline` instance.
        The shape of the inputs and outputs is the same as for
        `scipy.interpolate.RectBivariateSpline`."""
        # the following code is taken from scipy/interpolate/interpolate.py
        if self.bounds_error or self.fill_value is not None:
            out_of_bounds_x = (x < self.x_min) | (x > self.x_max)
            out_of_bounds_y = (y < self.y_min) | (y > self.y_max)

            any_out_of_bounds_x = np.any(out_of_bounds_x)
            any_out_of_bounds_y = np.any(out_of_bounds_y)

        if self.bounds_error and (any_out_of_bounds_x or any_out_of_bounds_y):
            raise ValueError("Values out of range; x must be in %r, y in %r"
                             % ((self.x_min, self.x_max),
                                (self.y_min, self.y_max)))

        z = super().__call__(x, y, *args, **kwargs)

        if self.fill_value is not None:
            # print('fill value is not None')
            if any_out_of_bounds_x:
                # print(f'some x is out of bound: {out_of_bounds_x}')
                # print(f'x is {x}')
                # print(f'x_min is {self.x_min}')
                # print(f'x_max is {self.x_max}')
                z[out_of_bounds_x, :] = self.fill_value
            if any_out_of_bounds_y:
                # print(f'some y is out of bound: {out_of_bounds_y}')
                # print(f'y is {y}')
                # print(f'y_min is {self.y_min}')
                # print(f'y_max is {self.y_max}')
                z[:, out_of_bounds_y] = self.fill_value
        return z

parton.pdf.MyRectBivariateSpline = MyRectBivariateSpline


@lru_cache()
def get_pdf(name, member=0, download=False):
    try:
        pdf = parton.PDF(name, member)
    except ValueError as e:
        if "Data file" in str(e):
            raise Exception(f"PDF set '{name}' does not provide member '{member}'")
        elif download is True:
            dir = parton.io.data_dir()
            parton.io.download_pdfset(name, dir)
            pdf = parton.PDF(name, member)
        else:
            raise Exception(f"PDF set '{name}' not available. Please install the PDF set using the command \"python3 -m parton install '{name}'\"")
    return pdf

@lru_cache(maxsize=None)
def get_parton_lumi(Q2, member=0):
    pdfset = config['PDF set']['dileptons']
    pdf = get_pdf(pdfset, member)
    return parton.PLumi(pdf, Q2=Q2)
