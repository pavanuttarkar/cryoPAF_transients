import numpy as np


CAND_DTYPE_CLS_BEAM = np.dtype([
    ("sn",     "f8"),
    ("sampno", "f8"),
    ("secs",   "f8"),
    ("boxcar", "f8"),
    ("idt",    "f8"),
    ("dm",     "f8"),
    ("beamno", "f8"),
    ("mjd",    "f8"),
    ("cls",    "f8"),
    ("ncands", "f8"),
    ("beams", object),   
])

CAND_DTYPE_CLS = np.dtype([
    ("sn",     "f8"),
    ("sampno", "f8"),
    ("secs",   "f8"),
    ("boxcar", "f8"),
    ("idt",    "f8"),
    ("dm",     "f8"),
    ("beamno", "f8"),
    ("mjd",    "f8"),
    ("cls",    "f8"),
    ("ncands",    "f8"),
])



# Raw single-pulse candidate file:
#CAND_DTYPE_CLS = np.dtype([
#    ("sn",     "f8"),
#    ("sampno", "f8"),
#    ("secs",   "f8"),
#    ("boxcar", "f8"),
#    ("idt",    "f8"),
#    ("dm",     "f8"),
#    ("beamno", "f8"),
#    ("mjd",    "f8"),
#    ("cls",    "f8"),
#])

## S/N, sampno, secs from file start, boxcar, idt, dm, beamno, mjd
#CAND_DTYPE = np.dtype([
#    ("sn",     "f8"),
#    ("sampno", "i8"),
#    ("secs",   "f8"),
#    ("boxcar", "i8"),
#    ("idt",    "i8"),
#    ("dm",     "f8"),
#    ("beamno", "i8"),
#])


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


NEW_CAND_DTYPE = np.dtype([
    ("sn",         np.float64),
    ("sampno",     np.int64),
    ("secs",       np.float64),
    ("boxcar",     np.int32),
    ("idt",        np.int32),
    ("dm",         np.float64),
    ("beamno",     np.int32),
    ("mjd",        np.float64),   
    ("ra",         "U12"),        
    ("dec",        "U12"),        
    ("match_name", "U32"),        
    ("match_ra",   "U12"),
    ("match_dec",  "U12"),
    ("match_dm",   np.float64),
    ("sep_deg",    np.float64),
    ("ddm",        np.float64),
    ("ddm_frac",   np.float64),
    ("is_match",   np.bool_),
])


CAND_DTYPE_CLS_BEAM = np.dtype(NEW_CAND_DTYPE.descr + [("beams", "O")])
 
