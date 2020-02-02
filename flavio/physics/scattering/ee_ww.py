r"""Functions for $e^+ e^-\to W^+ W^- scattering"""

import flavio


def ee_WW(C, E):
    r"""$e^+e^-\to W^+W^-$ cross section normalized to the SM"""
    if E == 161.3:
        res = (-0.196 * C['phiWB']
              -0.088 * C['phiD']
              +0.069 * C['phil3_11']
              -0.175 * C['phil3_22']
              +0.087 * C['ll_1221']
              -0.001 * C['phil1_11']
              -0.001 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 172.1:
        res = (-0.001 * C['W']
              -0.186 * C['phiWB']
              -0.086 * C['phiD']
              +0.072 * C['phil3_11']
              -0.172 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.005 * C['phil1_11']
              -0.006 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 182.7:
        res = (-0.002 * C['W']
              -0.18 * C['phiWB']
              -0.085 * C['phiD']
              +0.076 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.009 * C['phil1_11']
              -0.009 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 188.6:
        res = (-0.002 * C['W']
              -0.178 * C['phiWB']
              -0.085 * C['phiD']
              +0.078 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.012 * C['phil1_11']
              -0.011 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 191.6:
        res = (-0.003 * C['W']
              -0.178 * C['phiWB']
              -0.086 * C['phiD']
              +0.079 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.013 * C['phil1_11']
              -0.012 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 195.5:
        res = (-0.003 * C['W']
              -0.177 * C['phiWB']
              -0.085 * C['phiD']
              +0.081 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.014 * C['phil1_11']
              -0.013 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 199.5:
        res = (-0.003 * C['W']
              -0.176 * C['phiWB']
              -0.085 * C['phiD']
              +0.082 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.085 * C['ll_1221']
              -0.016 * C['phil1_11']
              -0.013 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 201.6:
        res = (-0.004 * C['W']
              -0.176 * C['phiWB']
              -0.086 * C['phiD']
              +0.083 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.016 * C['phil1_11']
              -0.014 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 204.9:
        res = (-0.004 * C['W']
              -0.175 * C['phiWB']
              -0.086 * C['phiD']
              +0.084 * C['phil3_11']
              -0.17 * C['phil3_22']
              +0.085 * C['ll_1221']
              -0.018 * C['phil1_11']
              -0.014 * C['phie_11'])
        return 1 + 1e6 * res.real
    if E == 206.6:
        res = (-0.004 * C['W']
              -0.175 * C['phiWB']
              -0.086 * C['phiD']
              +0.085 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.018 * C['phil1_11']
              -0.015 * C['phie_11'])
        return 1 + 1e6 * res.real
    raise ValueError("The ee->WW cross section is not defined for {} GeV.".format(E))


def ee_WW_obs(wc_obj, par, E):
    scale = flavio.config['renormalization scale']['ee_ww']
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_WW(C, E)


def ee_WW_diff(C, E, thetamin, thetamax):
    if E == 182.66:
        if (thetamin, thetamax) == (-1, -0.8):
            res = 0.702 + 246.22**2 * (- 1.20275 * C['phiD']- 1.90731 * C['phiWB']- 0.627597 * C['W']-   0.282334 * C['phie_11'] - 1.50752 * C['phil1_11'] + 1.72957 * C['phil3_11'] -   1.68816 * C['phil3_22'] + 0.844082 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.8, -0.6):
            res = 0.841 + 246.22**2 * (- 1.29583 * C['phiD']- 2.1494 * C['phiWB']- 0.582821 * C['W']-   0.301465 * C['phie_11'] - 1.43009 * C['phil1_11'] + 1.84714 * C['phil3_11'] -   1.99663 * C['phil3_22'] + 0.998316 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.6, -0.4):
            res = 1.011 + 246.22**2 * (- 1.40852 * C['phiD']- 2.43772 * C['phiWB']- 0.527908 * C['W']-   0.315814 * C['phie_11'] - 1.33484 * C['phil1_11'] + 1.98417 * C['phil3_11'] -   2.36628 * C['phil3_22'] + 1.18314 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.4, -0.2):
            res = 1.181 + 246.22**2 * (- 1.53366 * C['phiD']- 2.76037 * C['phiWB']- 0.458975 * C['W']-   0.325379 * C['phie_11'] - 1.21185 * C['phil1_11'] + 2.15893 * C['phil3_11'] -   2.81657 * C['phil3_22'] + 1.40829 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.2, 0.0):
            res = 1.402 + 246.22**2 * (- 1.65497 * C['phiD']- 3.08765 * C['phiWB']- 0.369852 * C['W']-   0.330162 * C['phie_11'] - 1.0454 * C['phil1_11'] + 2.40228 * C['phil3_11'] -   3.37424 * C['phil3_22'] + 1.68712 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0, 0.2):
            res = 1.731 + 246.22**2 * (- 1.73519 * C['phiD']- 3.3478 * C['phiWB']- 0.250124 * C['W']-   0.330162 * C['phie_11'] - 0.808946 * C['phil1_11'] + 2.76975 * C['phil3_11'] -   4.07596 * C['phil3_22'] + 2.03798 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.2, 0.4):
            res = 2.189 + 246.22**2 * (- 1.68486 * C['phiD']- 3.36256 * C['phiWB']- 0.0806916 * C['W']-   0.325379 * C['phie_11'] - 0.453783 * C['phil1_11'] + 3.36979 * C['phil3_11'] -   4.96981 * C['phil3_22'] + 2.4849 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.4, 0.6):
            res = 2.822 + 246.22**2 * (- 1.27057 * C['phiD']- 2.65627 * C['phiWB']+ 0.177727 * C['W']-   0.315814 * C['phie_11'] + 0.120226 * C['phil1_11'] + 4.43694 * C['phil3_11'] -   6.10597 * C['phil3_22'] + 3.05298 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.6, 0.8):
            res = 3.806 + 246.22**2 * (+ 0.189999 * C['phiD']+ 0.180739 * C['phiWB']+ 0.621186 * C['W']-   0.301465 * C['phie_11'] + 1.15795 * C['phil1_11'] + 6.5407 * C['phil3_11'] -   7.46705 * C['phil3_22'] + 3.73352 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.8, 1.0):
            res = 5.434 + 246.22**2 * (+ 4.6912 * C['phiD']+ 9.23155 * C['phiWB']+ 1.56709 * C['W']-   0.282334 * C['phie_11'] + 3.46857 * C['phil1_11'] + 10.9466 * C['phil3_11'] -   8.69668 * C['phil3_22'] + 4.34834 * C['ll_1221'])
            return res.real
    if E == 189.09:
        if (thetamin, thetamax) == (-1, -0.8):
            res = 0.661 + 246.22**2 * (-   1.07736 * C['phiD']- 1.57947 * C['phiWB']- 0.666332 * C['W']- 0.339143 * C['phie_11'] -   1.59121 * C['phil1_11'] + 1.74493 * C['phil3_11'] - 1.46699 * C['phil3_22'] +   0.733497 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.8, -0.6):
            res = 0.781 + 246.22**2 * (- 1.20021 * C['phiD']- 1.88528 * C['phiWB']- 0.625057 * C['W']-   0.363508 * C['phie_11'] - 1.52971 * C['phil1_11'] + 1.85182 * C['phil3_11'] -   1.78514 * C['phil3_22'] + 0.892572 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.6, -0.4):
            res = 0.928 + 246.22**2 * (- 1.35429 * C['phiD']- 2.26048 * C['phiWB']- 0.574047 * C['W']-   0.381782 * C['phie_11'] - 1.45229 * C['phil1_11'] + 1.97097 * C['phil3_11'] -   2.16953 * C['phil3_22'] + 1.08476 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.4, -0.2):
            res = 1.137 + 246.22**2 * (- 1.54063 * C['phiD']- 2.71114 * C['phiWB']- 0.50939 * C['W']-   0.393964 * C['phie_11'] - 1.34913 * C['phil1_11'] + 2.11777 * C['phil3_11'] -   2.64662 * C['phil3_22'] + 1.32331 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.2, 0.0):
            res = 1.403 + 246.22**2 * (- 1.75649 * C['phiD']- 3.23742 * C['phiWB']- 0.424752 * C['W']-   0.400056 * C['phie_11'] - 1.20432 * C['phil1_11'] + 2.31874 * C['phil3_11'] -   3.25554 * C['phil3_22'] + 1.62777 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0, 0.2):
            res = 1.715 + 246.22**2 * (- 1.9878 * C['phiD']- 3.819 * C['phiWB']- 0.309149 * C['W']-   0.400056 * C['phie_11'] - 0.990249 * C['phil1_11'] + 2.62303 * C['phil3_11'] -   4.05586 * C['phil3_22'] + 2.02793 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.2, 0.4):
            res = 2.187 + 246.22**2 * (- 2.18582 * C['phiD']- 4.36863 * C['phiWB']- 0.141698 * C['W']-   0.393964 * C['phie_11'] - 0.654462 * C['phil1_11'] + 3.13082 * C['phil3_11'] -   5.14082 * C['phil3_22'] + 2.57041 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.4, 0.6):
            res = 2.946 + 246.22**2 * (- 2.18545 * C['phiD']- 4.56602 * C['phiWB']+ 0.122875 * C['W']-   0.381782 * C['phie_11'] - 0.0832134 * C['phil1_11'] + 4.07421 * C['phil3_11'] -   6.65501 * C['phil3_22'] + 3.3275 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.6, 0.8):
            res = 4.122 + 246.22**2 * (- 1.36585 * C['phiD']- 3.1581 * C['phiWB']+ 0.605199 * C['W']-   0.363508 * C['phie_11'] + 1.02656 * C['phil1_11'] + 6.08696 * C['phil3_11'] -   8.77845 * C['phil3_22'] + 4.38922 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.8, 1.0):
            res = 6.253 + 246.22**2 * (+ 2.71439 * C['phiD']+ 4.78193 * C['phiWB']+ 1.78025 * C['W']-   0.339143 * C['phie_11'] + 3.86816 * C['phil1_11'] + 10.8483 * C['phil3_11'] -   11.4077 * C['phil3_22'] + 5.70384 * C['ll_1221'])
            return res.real
    if E == 198.38:
        if (thetamin, thetamax) == (-1, -0.8):
            res = 0.542 + 246.22**2 * (-   0.891093 * C['phiD']- 1.13999 * C['phiWB']- 0.687801 * C['W']- 0.398332 * C['phie_11'] -   1.63042 * C['phil1_11'] + 1.73185 * C['phil3_11'] - 1.18765 * C['phil3_22'] +   0.593825 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.8, -0.6):
            res = 0.664 + 246.22**2 * (- 1.02746 * C['phiD']- 1.47321 * C['phiWB']- 0.652105 * C['W']-   0.429644 * C['phie_11'] - 1.59301 * C['phil1_11'] + 1.83664 * C['phil3_11'] -   1.49608 * C['phil3_22'] + 0.748039 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.6, -0.4):
            res = 0.835 + 246.22**2 * (- 1.20082 * C['phiD']- 1.88519 * C['phiWB']- 0.60767 * C['W']-   0.453128 * C['phie_11'] - 1.54211 * C['phil1_11'] + 1.94724 * C['phil3_11'] -   1.86891 * C['phil3_22'] + 0.934454 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.4, -0.2):
            res = 1.021 + 246.22**2 * (- 1.41938 * C['phiD']- 2.39746 * C['phiWB']- 0.55083 * C['W']-   0.468784 * C['phie_11'] - 1.46867 * C['phil1_11'] + 2.07634 * C['phil3_11'] -   2.33786 * C['phil3_22'] + 1.16893 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.2, 0.0):
            res = 1.265 + 246.22**2 * (- 1.6932 * C['phiD']- 3.03789 * C['phiWB']- 0.475536 * C['W']-   0.476613 * C['phie_11'] - 1.3578 * C['phil1_11'] + 2.24604 * C['phil3_11'] -   2.95221 * C['phil3_22'] + 1.4761 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0, 0.2):
            res = 1.666 + 246.22**2 * (- 2.0323 * C['phiD']- 3.8387 * C['phiWB']- 0.371028 * C['W']-   0.476613 * C['phie_11'] - 1.18294 * C['phil1_11'] + 2.49817 * C['phil3_11'] -   3.79255 * C['phil3_22'] + 1.89628 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.2, 0.4):
            res = 2.161 + 246.22**2 * (- 2.43623 * C['phiD']- 4.81872 * C['phiWB']- 0.216107 * C['W']-   0.468784 * C['phie_11'] - 0.891835 * C['phil1_11'] + 2.92119 * C['phil3_11'] -   4.99905 * C['phil3_22'] + 2.49953 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.4, 0.6):
            res = 3.003 + 246.22**2 * (- 2.84154 * C['phiD']- 5.88317 * C['phiWB']+ 0.0376931 * C['W']-   0.453128 * C['phie_11'] - 0.364955 * C['phil1_11'] + 3.73339 * C['phil3_11'] -   6.8322 * C['phil3_22'] + 3.4161 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.6, 0.8):
            res = 4.428 + 246.22**2 * (- 2.81406 * C['phiD']- 6.20575 * C['phiWB']+ 0.531897 * C['W']-   0.429644 * C['phie_11'] + 0.746691 * C['phil1_11'] + 5.61057 * C['phil3_11'] -   9.78752 * C['phil3_22'] + 4.89376 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.8, 1.0):
            res = 7.236 + 246.22**2 * (+ 0.48308 * C['phiD']- 0.169772 * C['phiWB']+ 1.95866 * C['W']-   0.398332 * C['phie_11'] + 4.14993 * C['phil1_11'] + 10.8574 * C['phil3_11'] -   14.4185 * C['phil3_22'] + 7.20925 * C['ll_1221'])
            return res.real
    if E == 205.92:
        if (thetamin, thetamax) == (-1, -0.8):
            res = 0.532 + 246.22**2 * (-   0.761008 * C['phiD']- 0.854014 * C['phiWB']- 0.687161 * C['W']- 0.430144 * C['phie_11'] -   1.62101 * C['phil1_11'] + 1.69882 * C['phil3_11'] - 1.00547 * C['phil3_22'] +   0.502733 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.8, -0.6):
            res = 0.642 + 246.22**2 * (- 0.897142 * C['phiD']- 1.18526 * C['phiWB']- 0.655711 * C['W']-   0.466585 * C['phie_11'] - 1.60237 * C['phil1_11'] + 1.80701 * C['phil3_11'] -   1.29823 * C['phil3_22'] + 0.649116 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.6, -0.4):
            res = 0.77 + 246.22**2 * (- 1.07023 * C['phiD']- 1.5929 * C['phiWB']- 0.616408 * C['W']-   0.493916 * C['phie_11'] - 1.57151 * C['phil1_11'] + 1.91745 * C['phil3_11'] -   1.65052 * C['phil3_22'] + 0.82526 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.4, -0.2):
            res = 0.972 + 246.22**2 * (- 1.29164 * C['phiD']- 2.1051 * C['phiWB']- 0.565886 * C['W']-   0.512137 * C['phie_11'] - 1.52021 * C['phil1_11'] + 2.04138 * C['phil3_11'] -   2.09565 * C['phil3_22'] + 1.04782 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (-0.2, 0.0):
            res = 1.231 + 246.22**2 * (- 1.57752 * C['phiD']- 2.76261 * C['phiWB']- 0.498529 * C['W']-   0.521247 * C['phie_11'] - 1.43482 * C['phil1_11'] + 2.19851 * C['phil3_11'] -   2.68666 * C['phil3_22'] + 1.34333 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0, 0.2):
            res = 1.561 + 246.22**2 * (- 1.95043 * C['phiD']- 3.6244 * C['phiWB']- 0.404205 * C['W']-   0.521247 * C['phie_11'] - 1.29067 * C['phil1_11'] + 2.42638 * C['phil3_11'] -   3.51324 * C['phil3_22'] + 1.75662 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.2, 0.4):
            res = 2.056 + 246.22**2 * (- 2.43792 * C['phiD']- 4.76978 * C['phiWB']- 0.262564 * C['W']-   0.512137 * C['phie_11'] - 1.03824 * C['phil1_11'] + 2.80557 * C['phil3_11'] -   4.73908 * C['phil3_22'] + 2.36954 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.4, 0.6):
            res = 2.903 + 246.22**2 * (- 3.04602 * C['phiD']- 6.25554 * C['phiWB']- 0.025645 * C['W']-   0.493916 * C['phie_11'] - 0.56081 * C['phil1_11'] + 3.54114 * C['phil3_11'] -   6.69335 * C['phil3_22'] + 3.34668 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.6, 0.8):
            res = 4.445 + 246.22**2 * (- 3.52096 * C['phiD']- 7.6554 * C['phiWB']+ 0.454436 * C['W']-   0.466585 * C['phie_11'] + 0.501528 * C['phil1_11'] + 5.31438 * C['phil3_11'] -   10.1079 * C['phil3_22'] + 5.05394 * C['ll_1221'])
            return res.real
        if (thetamin, thetamax) == (0.8, 1.0):
            res = 7.783 + 246.22**2 * (- 0.937901 * C['phiD']- 3.27287 * C['phiWB']+ 2.01826 * C['W']-   0.430144 * C['phie_11'] + 4.19449 * C['phil1_11'] + 10.9353 * C['phil3_11'] -   16.273 * C['phil3_22'] + 8.13651 * C['ll_1221'])
            return res.real
    raise ValueError("The ee->WW cross section is not defined for {}, {}, {}".format(E, thetamin, thetamax))

def ee_WW_diff_obs(wc_obj, par, E, thetamin, thetamax):
    scale = flavio.config['renormalization scale']['ee_ww']
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_WW_diff(C, E, thetamin, thetamax)


_process_tex = r"e^+e^- \to W^+W^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to VV$ :: $' + _process_tex + r"$"

_obs_name = "R(ee->WW)"
_obs = flavio.classes.Observable(_obs_name)
_obs.arguments = ['E']
flavio.classes.Prediction(_obs_name, ee_WW_obs)
_obs.set_description(r"Cross section of $" + _process_tex + r"$ at energy $E$ normalized to the SM")
_obs.tex = r"$R(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)

_obs_name = "<dR/dtheta>(ee->WW)"
_obs = flavio.classes.Observable(_obs_name)
_obs.arguments = ['E', 'thetamin', 'thetamax']
flavio.classes.Prediction(_obs_name, ee_WW_diff_obs)
_obs.set_description(r"Differential cross section of $" + _process_tex + r"$ at energy $E$ binned in angle $\theta$ normalized to the SM")
_obs.tex = r"$\left\langle\frac{dR}{d\theta}\right\rangle(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
