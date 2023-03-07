"""Functions for parton distributions."""


import parton
from functools import lru_cache


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
    pdf = get_pdf('NNPDF40_nnlo_as_01180', member)
    return parton.PLumi(pdf, Q2=Q2)
