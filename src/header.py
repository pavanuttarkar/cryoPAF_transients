import numpy as np

# Raw single-pulse candidate file:
CAND_DTYPE_CLS = np.dtype([
    ("sn",     "f8"),
    ("sampno", "i8"),
    ("secs",   "f8"),
    ("boxcar", "i8"),
    ("idt",    "i8"),
    ("dm",     "f8"),
    ("beamno", "i8"),
    ("mjd",    "f8"),
    ("cls",    "f8"),
])

# S/N, sampno, secs from file start, boxcar, idt, dm, beamno, mjd
CAND_DTYPE = np.dtype([
    ("sn",     "f8"),
    ("sampno", "i8"),
    ("secs",   "f8"),
    ("boxcar", "i8"),
    ("idt",    "i8"),
    ("dm",     "f8"),
    ("beamno", "i8"),
    ("mjd",    "f8"),
])

# Candidate file that already carries ra/dec instead of beamno
CAND_RADEC_DTYPE = np.dtype([
    ("sn",     "f8"),
    ("sampno", "i8"),
    ("secs",   "f8"),
    ("boxcar", "i8"),
    ("idt",    "i8"),
    ("dm",     "f8"),
    ("ra",     "U16"),
    ("dec",    "U16"),
    ("mjd",    "f8"),
])

BEAM_DTYPE = np.dtype([
    ("beamno", "i8"),
    ("ra",     "U16"),
    ("dec",    "U16"),
])

# Known-pulsar catalog (matches fetch_atnf_catalog.py's CATALOG_DTYPE)
CATALOG_DTYPE = np.dtype([
    ("name", "U32"),
    ("ra",   "U16"),
    ("dec",  "U16"),
    ("dm",   "f8"),
])

# Output: candidate fields + match info
MATCH_EXTRA_DTYPE = [
    ("match_name", "U32"),
    ("match_ra",   "U16"),
    ("match_dec",  "U16"),
    ("match_dm",   "f8"),
    ("sep_deg",    "f8"),
    ("ddm",        "f8"),
    ("ddm_frac",   "f8"),
    ("is_match",   "bool"),
]

