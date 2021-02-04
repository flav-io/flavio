import flavio
from flavio.classes import Observable, Prediction
import numpy as np
from flavio.physics.edms.common import proton_charges

r"""Functions for neutrinoless $\mu - e$ conversion in different target nuclei"""
def CR_mue(wc_obj, par, nucl):
    r"""Conversion rate independent of the target nucleus"""
    mm = par['m_mu']
    scale = flavio.config['renormalization scale']['mudecays']
    #####overlap integrals and other parameters#####
    pc = proton_charges(par, scale)
    GuSp = (pc['gS_u+d'] + pc['gS_u-d']) / 2
    GdSp = (pc['gS_u+d'] - pc['gS_u-d']) / 2
    GsSp = pc['gS_s']
    GuSn = GdSp
    GdSn = GuSp
    GsSn = GsSp
    D   = par['D ' +nucl]*mm**(5/2)
    Sp  = par['Sp '+nucl]*mm**(5/2)
    Vp  = par['Vp '+nucl]*mm**(5/2)
    Sn  = par['Sn '+nucl]*mm**(5/2)
    Vn  = par['Vn '+nucl]*mm**(5/2)
    omega_capt  = par['GammaCapture '+nucl]
    #####Wilson Coefficients######
    #####Conversion Rate obtained from hep-ph/0203110#####
    flavio.citations.register("Kitano:2002mt")
    wc = wc_obj.get_wc('mue', scale, par, nf_out=3)
    prefac = -np.sqrt(2)/par['GF']
    AL = prefac / ( 4 * mm ) * wc['Cgamma_emu']
    AR = prefac / ( 4 * mm ) * wc['Cgamma_mue'].conjugate()
    gRV = {
        'u': prefac * ( wc['CVRR_mueuu'] + wc['CVLR_uumue'] ) / 2,
        'd': prefac * ( wc['CVRR_muedd'] + wc['CVLR_ddmue'] ) / 2,
        's': prefac * ( wc['CVRR_muess'] + wc['CVLR_ssmue'] ) / 2,
    }
    gLV = {
        'u': prefac * ( wc['CVLR_mueuu'] + wc['CVLL_mueuu'] ) / 2,
        'd': prefac * ( wc['CVLR_muedd'] + wc['CVLL_muedd'] ) / 2,
        's': prefac * ( wc['CVLR_muess'] + wc['CVLL_muess'] ) / 2,
    }
    gRS = {
        'u': prefac * ( wc['CSRR_emuuu'] + wc['CSRL_emuuu'] ).conjugate() / 2,
        'd': prefac * ( wc['CSRR_emudd'] + wc['CSRL_emudd'] ).conjugate() / 2,
        's': prefac * ( wc['CSRR_emuss'] + wc['CSRL_emuss'] ).conjugate() / 2,
    }
    gLS = {
        'u': prefac * ( wc['CSRR_mueuu'] + wc['CSRL_mueuu'] ) / 2,
        'd': prefac * ( wc['CSRR_muedd'] + wc['CSRL_muedd'] ) / 2,
        's': prefac * ( wc['CSRR_muess'] + wc['CSRL_muess'] ) / 2,
    }
    lhc = (
        AR.conjugate() * D
        + ( 2 * gLV['u'] + gLV['d'] ) * Vp
        + ( gLV['u'] + 2 * gLV['d'] ) * Vn
        + ( GuSp * gLS['u'] + GdSp * gLS['d'] + GsSp * gLS['s'] ) * Sp
        + ( GuSn * gLS['u'] + GdSn * gLS['d'] + GsSn * gLS['s'] ) * Sn
    )
    rhc = (
        AL.conjugate() * D
        + ( 2 * gRV['u'] + gRV['d'] ) * Vp
        + ( gRV['u'] + 2 * gRV['d'] ) * Vn
        + ( GuSp * gRS['u'] + GdSp * gRS['d'] + GsSp * gRS['s'] ) * Sp
        + ( GuSn * gRS['u'] + GdSn * gRS['d'] + GsSn * gRS['s'] ) * Sn
    )
    omega_conv = 2 * par['GF']**2 * ( abs(lhc)**2 + abs(rhc)**2 )
    return omega_conv / omega_capt


def CR_mueAu(wc_obj, par):
    r"""Conversion rate for $\phantom k^{197}_{79} \mathrm{Au}$"""
    return CR_mue(wc_obj, par, 'Au')
def CR_mueAl(wc_obj, par):
    r"""Conversion rate for $\phantom k^{27}_{13} \mathrm{Al}$"""
    return CR_mue(wc_obj, par, 'Al')
def CR_mueTi(wc_obj, par):
    r"""Conversion rate for $\phantom k^{48}_{22} \mathrm{Ti}$"""
    return CR_mue(wc_obj, par, 'Ti')

CRAu = Observable('CR(mu->e, Au)')
Prediction('CR(mu->e, Au)', CR_mueAu)
CRAu.tex = r"$CR(\mu - e)$ in $\phantom k^{197}_{79} \mathrm{Au}$"
CRAu.description = r"Coherent conversion rate of $\mu^-$ to $e^-$ in $\phantom k^{197}_{79} \mathrm{Au}$"
CRAu.add_taxonomy(r'Process :: muon decays :: LFV decays :: $\mu N \to e N$ :: ' + CRAu.tex)

CRAl = Observable('CR(mu->e, Al)')
Prediction('CR(mu->e, Al)', CR_mueAl)
CRAl.tex = r"$CR(\mu - e)$ in $\phantom k^{27}_{13} \mathrm{Al}$"
CRAl.description = r"Coherent conversion rate of $\mu^-$ to $e^-$ in $\phantom k^{27}_{13} \mathrm{Al}$"
CRAl.add_taxonomy(r'Process :: muon decays :: LFV decays :: $\mu N \to e N$ :: ' + CRAl.tex)

CRTi = Observable('CR(mu->e, Ti)')
Prediction('CR(mu->e, Ti)', CR_mueTi)
CRTi.tex = r"$CR(\mu - e)$ in $\phantom k^{48}_{22} \mathrm{Ti}$"
CRTi.description = r"Coherent conversion rate of $\mu^-$ to $e^-$ in $\phantom k^{48}_{22} \mathrm{Ti}$"
CRTi.add_taxonomy(r'Process :: muon decays :: LFV decays :: $\mu N \to e N$ :: ' + CRTi.tex)
