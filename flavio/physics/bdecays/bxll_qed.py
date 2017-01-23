r"""Functions for logarithmically enhanced QED corrections to
$B\to X_q\ell^+\ell^-$ decays.

See arXiv:1503.04849."""


from flavio.math.functions import li2
from math import log, pi, sqrt
import numpy as np

QL = -1
QD = -1/3

def wem_22_HL_low(sh, mb, ml, scale, mc):
    return (((-838449149 / 243972822 - 17899229 / (1104572981 * sh) +
              (392862691 * sh) / 31182122 - (443267230 * sh**2) / 5952119 +
              (1449510493 * sh**3) / 4470323 - (1705073446 * sh**4) / 3641611) *
             log(mb**2 / ml**2)) / (2 * (1 - sh)**2) +
            (8 * (-181815993 / 80017184 - 12693855 / (544572539 * sh) +
                  (221147177 * sh) / 19116111 - (366087255 * sh**2) / 5620222 +
                  (2187873727 * sh**3) / 9738530 - (1146287631 * sh**4) / 3841850) *
             log(mb**2 / ml**2) * log(scale / 5)) / (9 * (1 - sh)**2) +
            (64 * (-211413383732 / 275091287823 - 28224354522 / (1561005617567 * sh) +
                   (6950607294150 * sh) / 2369393418113 - (4051634435005 * sh**2) /
                   190407815151 + (1369675782386 * sh**3) / 19543876333 -
                   (13743778030206 * sh**4) / 170081997521) * log(mb**2 / ml**2) *
             log(scale / 5)**2) / (81 * (1 - sh)**2))


def wem_22_HT_low(sh, mb, ml, scale, mc):
    return (((790815147 / 139102337 + 5851171 / (263285345 * sh) -
              (385705146 * sh) / 23378083 + (1289689449 * sh**2) / 13308520 -
              (2659127785 * sh**3) / 6837791 + (5074697501 * sh**4) / 9398507) *
             log(mb**2 / ml**2)) / (4 * (1 - sh)**2 * sh) +
            (4 * (556023927 / 122276516 + 23492836 / (632635989 * sh) -
                  (538773306 * sh) / 27369085 + (668306834 * sh**2) / 8367293 -
                  (1449288445 * sh**3) / 5613183 + (1638642955 * sh**4) / 4962842) *
             log(mb**2 / ml**2) * log(scale / 5)) / (9 * (1 - sh)**2 * sh) +
            (32 * (294285468141 / 130238056721 + 61882902450 / (2178598483177 * sh) -
                   (891893361745 * sh) / 65882339941 + (6721380600018 * sh**2) /
                   127078889453 - (2229129812537 * sh**3) / 15777931542 +
                   (1518960348015 * sh**4) / 9614649326) * log(mb**2 / ml**2) *
             log(scale / 5)**2) / (81 * (1 - sh)**2 * sh))


def wem_27_HL_low(sh, mb, ml, scale, mc):
    return wem_27_HL_low_re(sh, mb, ml, scale, mc) + 1j*wem_27_HL_low_im(sh, mb, ml, scale, mc)

def wem_27_HL_low_im(sh, mb, ml, scale, mc):
    return (((-80412997 / 37565951 + 13038283 / (898825811 * sh) +
              (364340337 * sh) / 29553346 - (981319301 * sh**2) / 7877073 +
              (956619253 * sh**3) / 1976499 - (1852642991 * sh**4) / 3146630) *
             log(mb**2 / ml**2)) / (4 * (1 - sh)**2))


def wem_27_HL_low_re(sh, mb, ml, scale, mc):
    return (((-108392357 / 13520587 - 17643820 / (136786007 * sh) +
              (235290566 * sh) / 4341613 - (844871355 * sh**2) / 3008096 +
              (12623723393 * sh**3) / 14301082 - (2437221703 * sh**4) / 2173903) *
             log(mb**2 / ml**2)) / (4 * (1 - sh)**2) +
            (2 * (-466068367933 / 77209498795 - 129273447062 / (677884726251 * sh) +
                  (802881229231 * sh) / 16844742882 - (2368358059956 * sh**2) / 8502168821 +
                  (25527275160892 * sh**3) / 31618628465 - (4733260375039 * sh**4) /
                  5278867011) * log(mb**2 / ml**2) * log(scale / 5)) / (9 * (1 - sh)**2))


def wem_27_HT_low(sh, mb, ml, scale, mc):
    return wem_27_HT_low_re(sh, mb, ml, scale, mc) + 1j*wem_27_HT_low_im(sh, mb, ml, scale, mc)


def wem_27_HT_low_im(sh, mb, ml, scale, mc):
    return (((335877285 / 134633864 - 6741599 / (578673197 * sh) -
              (224760229 * sh) / 22691017 + (77865509 * sh**2) / 661654 -
              (3067267360 * sh**3) / 6714221 + (14977927977 * sh**4) / 25030972) *
             log(mb**2 / ml**2)) / (8 * (1 - sh)**2))


def wem_27_HT_low_re(sh, mb, ml, scale, mc):
    return (((265357997 / 12325558 + 16220138 / (133610699 * sh) -
              (1342960989 * sh) / 7635931 + (5414469045 * sh**2) / 6194518 -
              (5933172597 * sh**3) / 2314889 + (4917421927 * sh**4) / 1614947) *
             log(mb**2 / ml**2)) / (8 * (1 - sh)**2) +
            ((665193822413 / 34894567845 + 367186050192 / (1132464224467 * sh) -
              (8490574924798 * sh) / 45618896773 + (27892478448149 * sh**2) /
                33583938155 - (47591289842462 * sh**3) / 23069843393 +
                (34357103462268 * sh**4) / 15920585005) * log(mb**2 / ml**2) *
             log(scale / 5)) / (9 * (1 - sh)**2))

def wem_29_HL_low(sh, mb, ml, scale, mc):
    return wem_29_HL_low_re(sh, mb, ml, scale, mc) + 1j*wem_29_HL_low_im(sh, mb, ml, scale, mc)


def wem_29_HL_low_im(sh, mb, ml, scale, mc):
    return (((-106931022 / 160519171 + 37709726 / (13478294487 * sh) +
              (116283841 * sh) / 47554516 - (63732829 * sh**2) / 2254164 +
              (681786049 * sh**3) / 6236904 - (1091929622 * sh**4) / 9076491) *
             log(mb**2 / ml**2)) / (1 - sh)**2)


def wem_29_HL_low_re(sh, mb, ml, scale, mc):
    return (((-181815993 / 80017184 - 12693855 / (544572539 * sh) +
              (221147177 * sh) / 19116111 - (366087255 * sh**2) / 5620222 +
              (2187873727 * sh**3) / 9738530 - (1146287631 * sh**4) / 3841850) *
             log(mb**2 / ml**2)) / (1 - sh)**2 +
            (16 * (-211413383732 / 275091287823 - 28224354522 / (1561005617567 * sh) +
                   (6950607294150 * sh) / 2369393418113 - (4051634435005 * sh**2) /
                   190407815151 + (1369675782386 * sh**3) / 19543876333 -
                   (13743778030206 * sh**4) / 170081997521) * log(mb**2 / ml**2) *
             log(scale / 5)) / (9 * (1 - sh)**2))


def wem_29_HT_low(sh, mb, ml, scale, mc):
    return wem_29_HT_low_re(sh, mb, ml, scale, mc) + 1j*wem_29_HT_low_im(sh, mb, ml, scale, mc)


def wem_29_HT_low_im(sh, mb, ml, scale, mc):
    return (((769117403 / 1083206994 - (39752049 * sh) / 388465129 +
              (501354679 * sh**2) / 34218202 - (2114891079 * sh**3) / 34594484 +
              (143283798 * sh**4) / 1938497) * log(mb**2 / ml**2)) /
            (2 * (1 - sh)**2 * sh))


def wem_29_HT_low_re(sh, mb, ml, scale, mc):
    return (((556023927 / 122276516 + 23492836 / (632635989 * sh) -
              (538773306 * sh) / 27369085 + (668306834 * sh**2) / 8367293 -
              (1449288445 * sh**3) / 5613183 + (1638642955 * sh**4) / 4962842) *
             log(mb**2 / ml**2)) / (2 * (1 - sh)**2 * sh) +
            (8 * (294285468141 / 130238056721 + 61882902450 / (2178598483177 * sh) -
                  (891893361745 * sh) / 65882339941 + (6721380600018 * sh**2) /
                  127078889453 - (2229129812537 * sh**3) / 15777931542 +
                  (1518960348015 * sh**4) / 9614649326) * log(mb**2 / ml**2) *
             log(scale / 5)) / (9 * (1 - sh)**2 * sh))


def wem_77_HL_low(sh, mb, ml, scale, mc):
    return (((4577511406902 / 470085912983 - 489556167111 / (292184902699 * sh) -
              (2217854507559 * sh) / 32445538037 + (10040525817219 * sh**2) /
              36268506758 - (15225241057861 * sh**3) / 23691821747 +
              (10004690529043 * sh**4) / 15445369376) * log(mb**2 / ml**2)) /
            (4 * (1 - sh)**2))


def wem_77_HT_low(sh, mb, ml, scale, mc):
    return ((sh * (-1337100044249 / 37659770804 + 722187863398 / (465969559415 * sh) +
                   (7422422297404 * sh) / 41401596725 - (12586854481929 * sh**2) /
                   18412386451 + (10865783835307 * sh**3) / 6571860164 -
                   (29202520081607 * sh**4) / 17140449154) * log(mb**2 / ml**2)) /
            (8 * (1 - sh)**2))


def wem_79_HL_low(sh, mb, ml, scale, mc):
    return (((-466068367933 / 77209498795 - 129273447062 / (677884726251 * sh) +
              (802881229231 * sh) / 16844742882 - (2368358059956 * sh**2) / 8502168821 +
              (25527275160892 * sh**3) / 31618628465 - (4733260375039 * sh**4) /
              5278867011) * log(mb**2 / ml**2)) / (4 * (1 - sh)**2))


def wem_79_HT_low(sh, mb, ml, scale, mc):
    return (((665193822413 / 34894567845 + 367186050192 / (1132464224467 * sh) -
              (8490574924798 * sh) / 45618896773 + (27892478448149 * sh**2) /
              33583938155 - (47591289842462 * sh**3) / 23069843393 +
              (34357103462268 * sh**4) / 15920585005) * log(mb**2 / ml**2)) /
            (8 * (1 - sh)**2))


def wem_99_HL_low(sh, mb, ml, scale, mc):
    return (((-211413383732 / 275091287823 - 28224354522 / (1561005617567 * sh) +
              (6950607294150 * sh) / 2369393418113 - (4051634435005 * sh**2) /
              190407815151 + (1369675782386 * sh**3) / 19543876333 -
              (13743778030206 * sh**4) / 170081997521) * log(mb**2 / ml**2)) / (1 - sh)**2)


def wem_99_HT_low(sh, mb, ml, scale, mc):
    return (((294285468141 / 130238056721 + 61882902450 / (2178598483177 * sh) -
              (891893361745 * sh) / 65882339941 + (6721380600018 * sh**2) /
              127078889453 - (2229129812537 * sh**3) / 15777931542 +
              (1518960348015 * sh**4) / 9614649326) * log(mb**2 / ml**2)) /
            (2 * (1 - sh)**2 * sh))

@np.vectorize
def unit_step(x):
    if x >= 0:
        return 1
    elif x < 0:
        return 0

def wem_210_HA_low(sh, mb, ml, scale, mc):
    return (log(mb**2/ml**2)*(-(-(12921704/42341767) - 49563050/2047247 * sh
    + 278908189/1741458 * sh**2 - 218213887/577021 * sh**3 + 201127053/572486 * sh**4)/(24 * sh * (1 - sh)**2))
    + 8/9 * log(scale/ 5)*(log(mb**2/ ml**2) * (-((5 - 16 * sqrt(sh) + 11 * sh)/( 4 * (1 - sh)))
    + log(1 - sqrt(sh)) + ((1 - 5 * sh) * log( 1/2 * (1 + sqrt(sh))))/( 1 - sh) - ((1 - 3 * sh) * log(sh))/(1 - sh)))
    + 1j*(log(mb**2/ ml**2)*(-(-(66309793/8302997) - 246272299/1032560*(sh - (4 * mc**2/mb**2)**2)
    + 354752063/ 462598*(sh - (4 * mc**2/mb**2)**2)**2)/(24 * sh * (1 - sh)**2))*(sh - (4 * mc**2/mb**2)**2)**2* unit_step(sh - (4 * mc**2/mb**2)**2)))

def wem_710_HA_low(sh, mb, ml, scale, mc):
    return (log(mb**2/ml**2) * ((7 - 16 * sqrt(sh) + 9 * sh)/( 4 * (1 - sh))
    + log(1 - sqrt(sh)) + ((1 + 3 * sh) * log( 1/2 * (1 + sqrt(sh))))/(1 - sh) - (sh * log(sh))/( 1 - sh)))

def wem_910_HA_low(sh, mb, ml, scale, mc):
    return (log(mb**2/ml**2) * (-((5 - 16 * sqrt(sh) + 11 * sh)/( 4 * (1 - sh)))
    + log(1 - sqrt(sh)) + ((1 - 5 * sh) * log( 1/2 * (1 + sqrt(sh))))/( 1 - sh)
    - ((1 - 3 * sh) * log(sh))/(1 - sh)))


def wem_22_high(sh, mb, ml, scale, mc):
    return (((-1097570713 / 4986671 + (1177568126 * (1 - sh)) / 1344711 -
              (1556319889 * (1 - sh)**2) / 810348 + (743874878 * (1 - sh)**3) / 408259) *
             log(mb**2 / ml**2)) / (8 * (1 + 2 * sh)) +
            ((-278748564 / 1813913 + (725609439 * (1 - sh)) / 1454642 -
              (3679136911 * (1 - sh)**2) / 3208349 + (3067453889 * (1 - sh)**3) /
              2693552) * log(mb**2 / ml**2) * log(scale / 5)) / (9 * (1 + 2 * sh)) +
            (32 * QL**2 * log(mb**2 / ml**2) * log(scale / 5)**2 *
             (-1 - 3 * sh + 12 * sh**2 - 8 * sh**3 + 6 * (-1 + sh)**2 *
              (1 + 2 * sh) * log(1 - sh) - 3 * (1 - 6 * sh**2 + 4 * sh**3) *
                log(sh))) / (243 * (-1 + sh)**2 * (1 + 2 * sh)))

def wem_22_low(sh, mb, ml, scale, mc):
    return (((89893005 / 10186064 + 759773 / (27749657 * sh) - (74012480 * sh) / 6130973 +
              (46465860 * sh**2) / 749363 - (64848126 * sh**3) / 385135 +
              (78573521 * sh**4) / 422377) * log(mb**2 / ml**2)) /
            (8 * (1 - sh)**2 * (1 + 2 * sh)) +
            ((154715658 / 8534309 + 12910827 / (114675722 * sh) -
              (303518651 * sh) / 4850674 + (492828373 * sh**2) / 5448014 -
                (456369038 * sh**3) / 3593685) * log(mb**2 / ml**2) * log(scale / 5)) /
            (9 * (1 - sh)**2 * (1 + 2 * sh)) +
            (32 * QL**2 * log(mb**2 / ml**2) * log(scale / 5)**2 *
             (-1 - 3 * sh + 12 * sh**2 - 8 * sh**3 + 6 * (-1 + sh)**2 *
              (1 + 2 * sh) * log(1 - sh) - 3 * (1 - 6 * sh**2 + 4 * sh**3) *
                log(sh))) / (243 * (-1 + sh)**2 * (1 + 2 * sh)))



def wem_27_high(sh, mb, ml, scale, mc):
    return wem_27_re_high(sh, mb, ml, scale, mc) + 1j*wem_27_im_high(sh, mb, ml, scale, mc)


def wem_27_re_high(sh, mb, ml, scale, mc):
    return (((-1569999625 / 5062667 + (501693893 * (1 - sh)) / 601369 -
              (4187567828 * (1 - sh)**2) / 1919191 + (3462037641 * (1 - sh)**3) /
              1622488) * log(mb**2 / ml**2)) / 96 +
            (8 * log(mb**2 / ml**2) * log(scale / 5) * (1 / (2 * (-1 + sh)) +
                                                        log(1 - sh) - ((1 - 2 * sh + 2 * sh**2) * log(sh)) /
                                                        (2 * (-1 + sh)**2))) / 9)


def wem_27_im_high(sh, mb, ml, scale, mc):
    return (((-612530674 / 1182081 + (1248725501 * (1 - sh)) / 609972 -
              (12404910453 * (1 - sh)**2) / 2775125 + (3944570424 * (1 - sh)**3) / 817063) *
             log(mb**2 / ml**2)) / 96)


def wem_27_low(sh, mb, ml, scale, mc):
    return wem_27_re_low(sh, mb, ml, scale, mc) + 1j*wem_27_im_low(sh, mb, ml, scale, mc)

def wem_27_im_low(sh, mb, ml, scale, mc):
    return (((381643834 / 105094507 + (1368925501 * sh) / 123691778 -
              (1282659695 * sh**2) / 57060084 + (5654586858 * sh**3) / 29025721) *
             log(mb**2 / ml**2)) / (96 * (1 - sh)**2))



def wem_27_re_low(sh, mb, ml, scale, mc):
    return (((210088143 / 2815255 + 66788356 / (89691595 * sh) -
              (853046026 * sh) / 1861851 + (112055731 * sh**2) / 104256 -
              (534648229 * sh**3) / 441214) * log(mb**2 / ml**2)) / (96 * (1 - sh)**2) +
            (8 * log(mb**2 / ml**2) * log(scale / 5) * (1 / (2 * (-1 + sh)) +
                                                        log(1 - sh) - ((1 - 2 * sh + 2 * sh**2) * log(sh)) /
                                                        (2 * (-1 + sh)**2))) / 9)


def wem_29_high(sh, mb, ml, scale, mc):
    return wem_29_re_high(sh, mb, ml, scale, mc) + 1j*wem_29_re_high(sh, mb, ml, scale, mc)


def wem_29_re_high(sh, mb, ml, scale, mc):
    return (((-278748564 / 1813913 + (725609439 * (1 - sh)) / 1454642 -
              (3679136911 * (1 - sh)**2) / 3208349 + (3067453889 * (1 - sh)**3) /
              2693552) * log(mb**2 / ml**2)) / (8 * (1 + 2 * sh)) +
            (8 * QL**2 * log(mb**2 / ml**2) * log(scale / 5) * (-1 - 3 * sh + 12 * sh**2 -
                                                                8 * sh**3 + 6 * (-1 + sh)**2 * (1 + 2 * sh) * log(1 - sh) -
                                                                3 * (1 - 6 * sh**2 + 4 * sh**3) * log(sh))) /
            (27 * (-1 + sh)**2 * (1 + 2 * sh)))



def wem_29_im_high(sh, mb, ml, scale, mc):
    return (((-245650075 / 960653 + (817116203 * (1 - sh)) / 717333 -
              (1031991377 * (1 - sh)**2) / 427466 + (639292733 * (1 - sh)**3) / 268621) *
             log(mb**2 / ml**2)) / (8 * (1 + 2 * sh)))


def wem_29_im_low(sh, mb, ml, scale, mc):
    return (((243891481 / 182956795 + (371106943 * sh) / 94992737 -
              (607675179 * sh**2) / 82228837 + (8495644071 * sh**3) / 126392279) *
             log(mb**2 / ml**2)) / (8 * (1 - sh)**2 * (1 + 2 * sh)))


def wem_29_low(sh, mb, ml, scale, mc):
    return wem_29_re_low(sh, mb, ml, scale, mc) + 1j*wem_29_re_low(sh, mb, ml, scale, mc)

def wem_29_re_low(sh, mb, ml, scale, mc):
    return (((154715658 / 8534309 + 12910827 / (114675722 * sh) -
              (303518651 * sh) / 4850674 + (492828373 * sh**2) / 5448014 -
              (456369038 * sh**3) / 3593685) * log(mb**2 / ml**2)) /
            (8 * (1 - sh)**2 * (1 + 2 * sh)) +
            (8 * QL**2 * log(mb**2 / ml**2) * log(scale / 5) * (-1 - 3 * sh + 12 * sh**2 -
                                                                8 * sh**3 + 6 * (-1 + sh)**2 * (1 + 2 * sh) * log(1 - sh) -
                                                                3 * (1 - 6 * sh**2 + 4 * sh**3) * log(sh))) /
            (27 * (-1 + sh)**2 * (1 + 2 * sh)))

def wem_77_high(sh, mb, ml, scale, mc):
    return log(mb**2/ml**2)*(-sh/(2.*(-1 + sh)*(2 + sh)) + log(1 - sh) - (sh*(-3 + 2*sh**2)*log(sh))/(2.*(-1 + sh)**2*(2 + sh)))

def wem_99_high(sh, mb, ml, scale, mc):
    return (-0.5 + (log(mb**2/ml**2)*(-1 - 3*sh + 12*sh**2 - 8*sh**3 + 6*(-1 + sh)**2*(1 + 2*sh)*log(1 - sh) - 3*(1 - 6*sh**2 + 4*sh**3)*log(sh)))/(6.*(-1 + sh)**2*(1 + 2*sh)) +
  (-1 + 2*pi**2 + 6*sh - 15*sh**2 - 6*pi**2*sh**2 + 10*sh**3 + 4*pi**2*sh**3 + 12*(-1 + sh)**2*(1 + 2*sh)*log(1 - sh)*(-1 + log(sh)) + 4*log(sh) - 6*sh*log(sh) - 12*sh**2*log(sh) +
     8*sh**3*log(sh) - 6*log(sh)**2 + 36*sh**2*log(sh)**2 - 24*sh**3*log(sh)**2)/(12.*(-1 + sh)**2*(1 + 2*sh)) +
  (15 - 4*pi**2 + 12*sh - 45*sh**2 + 12*pi**2*sh**2 + 18*sh**3 - 8*pi**2*sh**3 - 12*sh*log(sh) + 12*sh**2*log(sh) + 24*sh**3*log(sh) -
     6*(-1 + sh)**2*log(1 - sh)*(5 + 4*sh + (2 + 4*sh)*log(sh)) - 24*(-1 + sh)**2*(1 + 2*sh)*li2(sh))/(216.*(-1 + sh)**2*(1 + 2*sh)))

def wem_79_high(sh, mb, ml, scale, mc):
    return log(mb**2/ml**2)*(1/(2.*(-1 + sh)) + log(1 - sh) - ((1 - 2*sh + 2*sh**2)*log(sh))/(2.*(-1 + sh)**2))

def wem_99_low(sh, mb, ml, scale, mc):
    return ()

def wem_99_low(sh, mb, ml, scale, mc):
    return (-0.5 + (log(mb**2/ml**2)*(-1 - 3*sh + 12*sh**2 - 8*sh**3 + 6*(-1 + sh)**2*(1 + 2*sh)*log(1 - sh) - 3*(1 - 6*sh**2 + 4*sh**3)*log(sh)))/(6.*(-1 + sh)**2*(1 + 2*sh)) +
    (-1 + 2*pi**2 + 6*sh - 15*sh**2 - 6*pi**2*sh**2 + 10*sh**3 + 4*pi**2*sh**3 + 12*(-1 + sh)**2*(1 + 2*sh)*log(1 - sh)*(-1 + log(sh)) + 4*log(sh) - 6*sh*log(sh) - 12*sh**2*log(sh) +
        8*sh**3*log(sh) - 6*log(sh)**2 + 36*sh**2*log(sh)**2 - 24*sh**3*log(sh)**2)/(12.*(-1 + sh)**2*(1 + 2*sh)) +
     (15 - 4*pi**2 + 12*sh - 45*sh**2 + 12*pi**2*sh**2 + 18*sh**3 - 8*pi**2*sh**3 - 12*sh*log(sh) + 12*sh**2*log(sh) + 24*sh**3*log(sh) -
        6*(-1 + sh)**2*log(1 - sh)*(5 + 4*sh + (2 + 4*sh)*log(sh)) - 24*(-1 + sh)**2*(1 + 2*sh)*li2(sh))/(216.*(-1 + sh)**2*(1 + 2*sh)))

def wem_77_low(sh, mb, ml, scale, mc):
    return (log(mb**2/ml**2)*(-sh/(2.*(-1 + sh)*(2 + sh)) + log(1 - sh) - (sh*(-3 + 2*sh**2)*log(sh))/(2.*(-1 + sh)**2*(2 + sh))))

def wem_79_low(sh, mb, ml, scale, mc):
    return (log(mb**2/ml**2)*(1/(2.*(-1 + sh)) + log(1 - sh) - ((1 - 2*sh + 2*sh**2)*log(sh))/(2.*(-1 + sh)**2)))

wem_dict_low = {}
wem_dict_low['L', 2, 2] = wem_22_HL_low
wem_dict_low['T', 2, 2] = wem_22_HT_low
wem_dict_low['L', 2, 7] = wem_27_HL_low
wem_dict_low['T', 2, 7] = wem_27_HT_low
wem_dict_low['L', 2, 9] = wem_29_HL_low
wem_dict_low['T', 2, 9] = wem_29_HT_low
wem_dict_low['L', 7, 7] = wem_77_HL_low
wem_dict_low['T', 7, 7] = wem_77_HT_low
wem_dict_low['L', 7, 9] = wem_79_HL_low
wem_dict_low['T', 7, 9] = wem_79_HT_low
wem_dict_low['L', 9, 9] = wem_99_HL_low
wem_dict_low['T', 9, 9] = wem_99_HT_low
wem_dict_low['BR', 2, 2] = wem_22_low
wem_dict_low['BR', 2, 7] = wem_27_low
wem_dict_low['BR', 2, 9] = wem_29_low
wem_dict_low['BR', 7, 7] = wem_77_low
wem_dict_low['BR', 7, 9] = wem_79_low
wem_dict_low['BR', 9, 9] = wem_99_low
wem_dict_low['A', 2, 10] = wem_210_HA_low
wem_dict_low['A', 7, 10] = wem_710_HA_low
wem_dict_low['A', 9, 10] = wem_910_HA_low
wem_dict_high = {}
wem_dict_high['BR', 2, 2] = wem_22_high
wem_dict_high['BR', 2, 7] = wem_27_high
wem_dict_high['BR', 2, 9] = wem_29_high
wem_dict_high['BR', 7, 7] = wem_77_high
wem_dict_high['BR', 7, 9] = wem_79_high
wem_dict_high['BR', 9, 9] = wem_99_high

def wem(I, i, j, sh, mb, ml, scale, mc):
    if sh < 0.5:
        return wem_dict_low[I, i, j](sh, mb, ml, scale, mc)
    else:
        return wem_dict_high[I, i, j](sh, mb, ml, scale, mc)
