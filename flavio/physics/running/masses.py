"""Functions for running of quark masses.

This module is based on the formulas in the `RunDec` papers, arXiv:hep-ph/0004189 and arXiv:1201.6149"""

from math import log, pi
import numpy as np
from flavio.math.functions import zeta
from flavio.physics.running.masses import zeta


def gamma0_qcd(nf):
    return 1.0
def gamma1_qcd(nf):
    return (202./3.-20.*nf/9.)/16.
def gamma2_qcd(nf):
    return (1249. + (-2216./27. - 160.*zeta(3)/3.)*nf-140.*nf*nf/81.)/64.
def gamma3_qcd(nf):
    return ((4603055./162. + 135680.*zeta(3)/27. - 8800.*zeta(5) +
          (-91723./27. - 34192.*zeta(3)/9. +
          880.*zeta(4) + 18400.*zeta(5)/9.)*nf +
          (5242./243. + 800.*zeta(3)/9. - 160.*zeta(4)/3.)*nf**2 +
          (-332./243. + 64.*zeta(3)/27.)*nf**3)/256.)

def gamma_qcd(mq, als, mu, f):
    r"""RHS of the QCD gamma function written in the (unconventional) form
    $d/d\mu m = gamma(\mu)$
    """
    g0 = gamma0_qcd(f)*(als/pi)**1
    g1 = gamma1_qcd(f)*(als/pi)**2
    g2 = gamma2_qcd(f)*(als/pi)**3
    g3 = gamma3_qcd(f)*(als/pi)**4
    return -2*mq/mu*(g0 + g1 + g2 + g3)

# OS to MSbar conversion according to RunDec

cf=4/3.
ca=3.
tr=1/2.
A4=0.5174790616738993863307581618988629456223774751413792582443193479770

def fMsFromOs1(mu, M):
     lmM=log((mu*mu)/(M*M))
     erg=  (-cf - (3.*cf*lmM)/4.)
     return erg

def fMsFromOs2(mu, M, nl):
     lmM=log((mu*mu)/(M*M))
     erg=  ((-1111.*ca*cf)/384. + (7.*cf*cf)/128. -
     (185.*ca*cf*lmM)/96. + (21.*cf*cf*lmM)/32. - (11.*ca*cf*lmM*lmM)/32. +
     (9.*cf*cf*lmM*lmM)/32. + (143.*cf*tr)/96. + (13.*cf*lmM*tr)/24. +
     (cf*lmM*lmM*tr)/8. +
     (71.*cf*nl*tr)/96. + (13.*cf*lmM*nl*tr)/24. + (cf*lmM*lmM*nl*tr)/8. +
     (ca*cf*zeta(2))/2 - (15.*cf*cf*zeta(2))/8. - (3.*ca*cf*log(2)*zeta(2))/2. +
     3.*cf*cf*log(2)*zeta(2) -
     cf*tr*zeta(2) + (cf*nl*tr*zeta(2))/2. + (3.*ca*cf*zeta(3))/8. -
     (3.*cf*cf*zeta(3))/4.)
     return erg

def fMsFromOs3(mu, M, nl):
     lmM=log((mu*mu)/(M*M))
     erg=  ((lmM*lmM*(-2341.*ca*ca*cf + 1962.*ca*cf*cf - 243.*cf*cf*cf
     + 1492.*ca*cf*tr -
      468.*cf*cf*tr + 1492.*ca*cf*nl*tr - 468.*cf*cf*nl*tr - 208.*cf*tr*tr -
      416.*cf*nl*tr*tr - 208.*cf*nl*nl*tr*tr))/1152. +
     (lmM*lmM*lmM*(-242.*ca*ca*cf + 297.*ca*cf*cf - 81.*cf*cf*cf +
     176.*ca*cf*tr - 108.*cf*cf*tr + 176.*ca*cf*nl*tr - 108.*cf*cf*nl*tr -
     32.*cf*tr*tr - 64.*cf*nl*tr*tr - 32.*cf*nl*nl*tr*tr))/1152. +
     (lmM*(-105944.*ca*ca*cf + 52317.*ca*cf*cf - 13203.*cf*cf*cf +
     74624.*ca*cf*tr -
     5436.*cf*cf*tr + 55616.*ca*cf*nl*tr + 2340.*cf*cf*nl*tr -
     12608.*cf*tr*tr -
     18304.*cf*nl*tr*tr - 5696.*cf*nl*nl*tr*tr + 12672.*ca*ca*cf*zeta(2) -
     52704.*ca*cf*cf*zeta(2) + 19440.*cf*cf*cf*zeta(2) -
     38016.*ca*ca*cf*log(2)*zeta(2) +
     91584.*ca*cf*cf*log(2)*zeta(2) - 31104.*cf*cf*cf*log(2)*zeta(2) -
     29952.*ca*cf*tr*zeta(2) +
     27648.*cf*cf*tr*zeta(2) + 13824.*ca*cf*log(2)*tr*zeta(2) -
     27648.*cf*cf*log(2)*tr*zeta(2) +
     8064.*ca*cf*nl*tr*zeta(2) + 12096.*cf*cf*nl*tr*zeta(2) +
     13824.*ca*cf*log(2)*nl*tr*zeta(2) -
     27648.*cf*cf*log(2)*nl*tr*zeta(2) + 9216.*cf*tr*tr*zeta(2) +
     4608.*cf*nl*tr*tr*zeta(2) -
     4608.*cf*nl*nl*tr*tr*zeta(2) + 9504.*ca*ca*cf*zeta(3) -
     22896.*ca*cf*cf*zeta(3) +
     7776.*cf*cf*cf*zeta(3) + 6912.*ca*cf*tr*zeta(3) - 3456.*cf*cf*tr*zeta(3) +
     6912.*ca*cf*nl*tr*zeta(3) - 3456.*cf*cf*nl*tr*zeta(3)))/13824.)
     return erg

def fOsFromMs1(mu, M):
    lmM = log((mu*mu)/(M*M))
    return (cf + (3.*cf*lmM)/4.)

def fOsFromMs2(mu, M, nl):
    lmM=log((mu*mu)/(M*M))
    return ((1111.*ca*cf)/384. - (71.*cf*cf)/128. -
    (143.*cf*tr)/96. - (71.*cf*nl*tr)/96. +
    lmM*((185.*ca*cf)/96. - (9.*cf*cf)/32. - (13.*cf*tr)/24. -
    (13.*cf*nl*tr)/24.) +
    lmM*lmM*((11.*ca*cf)/32. + (9.*cf*cf)/32. - (cf*tr)/8. - (cf*nl*tr)/8.) -
    (ca*cf*zeta(2))/2. + (15.*cf*cf*zeta(2))/8. + (3.*ca*cf*log(2)*zeta(2))/2.
    - 3.*cf*cf*log(2)*zeta(2) +
    cf*tr*zeta(2) - (cf*nl*tr*zeta(2))/2. - (3.*ca*cf*zeta(3))/8. +
    (3.*cf*cf*zeta(3))/4.)

def fOsFromMs3(mu, M, nl):
     lmM=log((mu*mu)/(M*M))
     return (lmM*lmM*lmM*((121.*ca*ca*cf)/576. + (33.*ca*cf*cf)/128. +
     (9.*cf*cf*cf)/128. - (11.*ca*cf*tr)/72. - (3.*cf*cf*tr)/32. -
     (11.*ca*cf*nl*tr)/72. -
     (3.*cf*cf*nl*tr)/32. + (cf*tr*tr)/36. + (cf*nl*tr*tr)/18. +
     (cf*nl*nl*tr*tr)/36.) + lmM*lmM*((2341.*ca*ca*cf)/1152.
     + (21.*ca*cf*cf)/64. -
     (63.*cf*cf*cf)/128. - (373.*ca*cf*tr)/288. - (3.*cf*cf*tr)/32. -
     (373.*ca*cf*nl*tr)/288. - (3.*cf*cf*nl*tr)/32. + (13.*cf*tr*tr)/72. +
     (13.*cf*nl*tr*tr)/36. + (13.*cf*nl*nl*tr*tr)/72.) +
     lmM*((13243.*ca*ca*cf)/1728. - (4219.*ca*cf*cf)/1536. +
     (495.*cf*cf*cf)/512. -
     (583.*ca*cf*tr)/108. - (307.*cf*cf*tr)/384. - (869.*ca*cf*nl*tr)/216. -
     (91.*cf*cf*nl*tr)/384. + (197.*cf*tr*tr)/216. + (143.*cf*nl*tr*tr)/108. +
     (89.*cf*nl*nl*tr*tr)/216. - (11.*ca*ca*cf*zeta(2))/12. +
     (49.*ca*cf*cf*zeta(2))/16. +
     (45.*cf*cf*cf*zeta(2))/32. + (11.*ca*ca*cf*log(2)*zeta(2))/4. -
     (35.*ca*cf*cf*log(2)*zeta(2))/8. -
     (9.*cf*cf*cf*log(2)*zeta(2))/4. + (13.*ca*cf*tr*zeta(2))/6. -
     (cf*cf*tr*zeta(2))/2. -
     ca*cf*log(2)*tr*zeta(2) + 2.*cf*cf*log(2)*tr*zeta(2) -
     (7.*ca*cf*nl*tr*zeta(2))/12. -
     (13.*cf*cf*nl*tr*zeta(2))/8. - ca*cf*log(2)*nl*tr*zeta(2) +
     2.*cf*cf*log(2)*nl*tr*zeta(2) -
     (2.*cf*tr*tr*zeta(2))/3. - (cf*nl*tr*tr*zeta(2))/3. +
     (cf*nl*nl*tr*tr*zeta(2))/3. -
     (11.*ca*ca*cf*zeta(3))/16. + (35.*ca*cf*cf*zeta(3))/32. +
     (9.*cf*cf*cf*zeta(3))/16. -
     (ca*cf*tr*zeta(3))/2. + (cf*cf*tr*zeta(3))/4. - (ca*cf*nl*tr*zeta(3))/2. +
     (cf*cf*nl*tr*zeta(3))/4.))

def f1SFromMs1(mu, mMS, asmu, nl):
     lmm = log(mu*mu/(mMS*mMS))
     return asmu*mMS*(12. - 2.*asmu*pi + 9.*lmm)/(9.*pi)

def f1SFromMs2(mu, mMS, asmu, nl):
     lmm = log(mu*mu/(mMS*mMS))
     log34 = log((3.*mu)/(4.*asmu*mMS))
     ln2 = log(2)
     Zeta3 = zeta(3)
     return (asmu*mMS*(-(2.*asmu)/9. + asmu*asmu*(-291. + 22.*nl - 198.*log34 +
          12.*nl*log34 - 3.*(8. + 6.*lmm))/(81.*pi) + (4. + 3.*lmm)/(3.*pi) +
          (asmu*(2763. - 142.*nl + 96.*pi*pi - 16.*nl*pi*pi + 32.*pi*pi*ln2 +
          2036.*lmm - 104.*nl*lmm + 564.*lmm*lmm - 24.*nl*lmm*lmm - 48.*Zeta3))/(288.*pi*pi)))

def f1SFromMs3(mu, mMS, asmu, nl):
     lmm = log(mu*mu/(mMS*mMS))
     log34 = log((3.*mu)/(4.*asmu*mMS))
     ln2 = log(2)
     Zeta3 = zeta(3)
     Zeta5 = zeta(5)
     return (asmu*mMS*(-(2.*asmu)/9. + asmu*asmu*(-291. + 22.*nl -  198.*log34 +
          12.*nl*log34)/(81.*pi) + (4. + 3.*lmm)/(3.*pi) - (2.*asmu*asmu*(4. + 3.*lmm))/(27.*pi) +
          asmu*asmu*asmu*(-372. + 40.*nl - 792.*log34 +
          48.*nl*log34 - 279.*lmm + 30.*nl*lmm -
          594.*log34*lmm + 36.*nl*log34*lmm)/(243.*pi*pi)
          + (asmu*(2763. - 142.*nl + 96.*pi*pi - 16.*nl*pi*pi + 32.*pi*pi*ln2 + 2036.*lmm - 104.*nl*lmm +
          564.*lmm*lmm - 24.*nl*lmm*lmm - 48.*Zeta3))/(288.*pi*pi) + asmu*asmu*asmu*(-2763.+ 142.*nl -
          96.*pi*pi + 16.*nl*pi*pi - 32.*pi*pi*ln2 -
          2036.*lmm + 104.*nl*lmm - 564.*lmm*lmm +
          24.*nl*lmm*lmm + 48.*Zeta3)/(1296.*pi*pi)
          + asmu*asmu*asmu*(-129096. + 20316.*nl - 616.*nl*nl -
          11668.*pi*pi + 528.*nl*pi*pi - 16.*nl*nl*pi*pi + 243.*pi*pi*pi*pi -
          200232.*log34 + 27792.*nl*log34 - 864.*nl*nl*log34 -
          78408.*log34*log34 + 9504.*nl*log34*log34 -
          288.*nl*nl*log34*log34 - 59400.*Zeta3 + 8208.*nl*Zeta3 -
          192.*nl*nl*Zeta3)/(3888*pi*pi) +
          (asmu*asmu*(42314585. - 4636940.*nl + 47060.*nl*nl + 7834092.*pi*pi -
          713520.*nl*pi*pi + 18720.*nl*nl*pi*pi - 41700.*pi*pi*pi*pi + 14640.*nl*pi*pi*pi*pi -
          1656000.*pi*pi*ln2 - 63360.*nl*pi*pi*ln2 - 126720.*pi*pi*ln2*ln2 +
          11520.*nl*pi*pi*ln2*ln2 - 158400.*ln2*ln2*ln2*ln2 + 5760.*nl*ln2*ln2*ln2*ln2 +
          33620760.*lmm - 3723120.*nl*lmm + 64080.*nl*nl*lmm + 1010880.*pi*pi*lmm - 220320.*nl*pi*pi*lmm + 8640.*nl*nl*pi*pi*lmm +
          336960.*pi*pi*ln2*lmm - 17280.*nl*pi*pi*ln2*lmm + 11726100.*lmm*lmm - 1247400.*nl*lmm*lmm + 28080.*nl*nl*lmm*lmm +
          2009880.*lmm*lmm*lmm - 185760.*nl*lmm*lmm*lmm + 4320.*nl*nl*lmm*lmm*lmm - 3801600.*A4 + 138240.*nl*A4 + 1002240.*Zeta3 -
          1561680.*nl*Zeta3 + 60480.*nl*nl*Zeta3 - 1554120.*pi*pi*Zeta3 - 894240.*lmm*Zeta3 - 362880.*nl*lmm*Zeta3 + 4266000.*Zeta5))/
          (466560.*pi*pi*pi)))

def fZmM(nl):
    return (-9478333./93312. + 55.*log(2)*log(2)*log(2)*log(2)/162. +
            (-644201./6480. + 587.*log(2)/27. + 44.*log(2)*log(2)/27.)*zeta(2) -
            61.*zeta(3)/27. + 3475*zeta(4)/432. + 1439.*zeta(2)*zeta(3)/72. -
            1975.*zeta(5)/216. + 220.*A4/27. + nl*(246643./23328. -
            log(2)*log(2)*log(2)*log(2)/81. +(967./108. + 22.*log(2)/27. -
            4.*log(2)*log(2)/27.)*zeta(2) + 241.*zeta(3)/72. - 305.*zeta(4)/108. -
            8.*A4/27.) + nl*nl*(-2353./23328. - 13.*zeta(2)/54 - 7.*zeta(3)/54.))

def fZmInvM(nl):
    return (8481925./93312. +
       (137.*nl)/216. + (652841.*pi*pi)/38880. - (nl*pi*pi)/27. -
       (695.*pi*pi*pi*pi)/7776. - (575.*pi*pi*log(2))/162. -
       (22.*pi*pi*log(2)*log(2))/81. -
       (55.*log(2)*log(2)*log(2)*log(2))/162. - (220.*A4)/27. -
       nl*nl*(-2353./23328. - (13.*pi*pi)/324. - (7.*zeta(3))/54.) +
       (58.*zeta(3))/27. -
       (1439.*pi*pi*zeta(3))/432. - nl*(246643./23328. + (967.*pi*pi)/648. -
       (61.*pi*pi*pi*pi)/1944. + (11.*pi*pi*log(2))/81. -
       (2.*pi*pi*log(2)*log(2))/81. -
       log(2)*log(2)*log(2)*log(2)/81. - (8.*A4)/27. +
       (241.*zeta(3))/72.) +
       (1975.*zeta(5))/216.)

def mOS2mMS(mOS, Nf, asmu, Mu, nl):
    s = np.zeros(4)
    s[0]= 1.
    s[1]=asmu*(fMsFromOs1(Mu, mOS))/pi
    s[2]=asmu**2*fMsFromOs2(Mu, mOS, Nf-1)/pi**2 # omitting the fDelta piece
    s[3]=asmu**3*(fMsFromOs3(Mu, mOS,Nf-1)+ fZmM(Nf-1))/pi**3
    erg=0.0
    if(nl==0):
        erg=1
    else:
       erg=s[:nl+1].sum()
    return mOS*erg

def mMS2mOS(MS, Nf, asmu, Mu, nl):
    s = np.zeros(4)
    s[0]= 1.
    s[1]=asmu*(fOsFromMs1(Mu, MS))/pi
    s[2]=asmu**2*fOsFromMs2(Mu, MS, Nf-1)/pi**2 # omitting the fDelta piece
    s[3]=asmu**3*(fOsFromMs3(Mu, MS,Nf-1)+ fZmInvM(Nf-1))/pi**3
    erg=0.0
    if(nl==0):
        erg=1
    else:
       erg=s[:nl+1].sum()
    return MS*erg

def mMS2m1S(MS, Nf, asmu, Mu, nl):
    if nl == 0:
        return MS
    elif nl == 1:
        return MS + f1SFromMs1(Mu, MS, asmu, nl=Nf)
    elif nl == 2:
        return MS + f1SFromMs2(Mu, MS, asmu, nl=Nf)
    elif nl == 3:
        return MS + f1SFromMs3(Mu, MS, asmu, nl=Nf)

# (19) of arXiv:1107.3100v1
def fKsFromMs1(Mu, M, Nf):
    return -(4/3.)* (1- (4/3) * (Mu/M) - (Mu**2/(2*M**2)) )
def fKsFromMs2(Mu, M, Nf):
    b0 = 11 - 2*Nf/3.
    return ((((1/3.)*log(M/(2*Mu))+13/18.)*b0 - pi**2/3. + 23/18.)*Mu**2/M**2
            + (((8/9.)*log(M/(2*Mu))+64/27.)*b0 - 8*pi**2/9. + 92/27.) * Mu/M
            - (pi**2/12. + 71/96.)*b0
            + zeta(3)/6. - pi**2/9. * log(2) + 7*pi**2/12. + 23/72.
            )
# from (A.8) of hep-ph/0302262v1
def fKsFromMs3(Mu, M, Nf):
    b0 = 11 - 2*Nf/3.
    return -(b0/2.)**2*(2353/2592.+13/36.*pi**2+7/6.*zeta(3)
            -16/9.*Mu/M*((log(M/(2*Mu))+8/3.)**2+67/36.-pi**2/6.)
            -2/3.*Mu**2/M**2*((log(M/(2*Mu))+13/6.)**2+10/9.-pi**2/6.))

def mKS2mMS(M, Nf, asM, Mu, nl):
    s = np.zeros(4)
    s[0] = 1.
    s[1] = (asM/pi) * fKsFromMs1(Mu, M, Nf)
    s[2] = (asM/pi)**2 * fKsFromMs2(Mu, M, Nf)
    s[3] = (asM/pi)**3 * fKsFromMs3(Mu, M, Nf)
    r = s[:nl+1].sum()
    return M * r

def mMS2mKS(MS, Nf, asM, Mu, nl):
    def convert(M):
        s = np.zeros(4)
        s[0] = 1.
        s[1] = -(asM/pi) * fKsFromMs1(Mu, M, Nf)
        s[2] = -(asM/pi)**2 * fKsFromMs2(Mu, M, Nf)
        # properly invert the relation to O(asM**2)
        s[2] = s[2] + s[1]**2
        s[3] = -(asM/pi)**3 * fKsFromMs3(Mu, M, Nf)
        r = s[:nl+1].sum()
        return MS * r
    # iterate twice
    Mtmp = convert(MS)
    Mtmp = convert(Mtmp)
    return convert (Mtmp)
