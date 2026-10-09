"""
pulsar_crossmatch.py
=====================
------------------------------------------------
Matching criteria
-----------------------------------------------
A candidate is called a match to a known pulsar if:
  1. Angular separation between beam pointing and pulsar position
     <= --radius.
  2. |DM_candidate - DM_pulsar| / DM_pulsar <= --dm-tol (default: 10 DMunits ),
     <= --dm-abs-tol if provided.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    python pulsar_crossmatch.py \
        --candidates candidates.txt \
        --beams beam_positions.txt \
        --radius 0.5 --dm-tol 0.1 \
        --out matches.csv

    # or pull the catalog live from ATNF:
    python pulsar_crossmatch.py \
        --candidates candidates.txt \
        --beams beam_positions.txt \
        --use-atnf \
        --out matches.csv
"""

__author__      =   "Pavan Uttarkar and Claude"
__version__     =   "0.0.1"
__maintainer__  =   "Pavan Uttarkar"
__email__       =   "pavan.uttarkar@gmail.com"
__status__      =   "Development"
__project__     =   "cryoPAF transient searches"

import argparse
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from header import CAND_DTYPE, CAND_RADEC_DTYPE, BEAM_DTYPE
from header import CATALOG_DTYPE, MATCH_EXTRA_DTYPE, CAND_DTYPE_CLS


def load_candidates(path: str) -> np.ndarray:
    """
    Load a single-pulse candidate file with columns:
    S/N, sampno, secs from file start, boxcar, idt, dm, beamno, mjd

    Lines starting with '#' are treated as comments/header and skipped.
    """
    return np.loadtxt(path, comments="#", dtype=CAND_DTYPE_CLS)


def load_candidates_with_radec(path: str) -> np.ndarray:
    """
    Alternative loader if candidate file already has RA/Dec columns
    instead of beamno. Expects columns matching CAND_RADEC_DTYPE, in
    order: sn, sampno, secs, boxcar, idt, dm, ra, dec, mjd.
    Adapt column order/dtype as needed for your actual file.
    """
    return np.loadtxt(path, comments="#", dtype=CAND_RADEC_DTYPE)



def read_beam_positions(path: str) -> np.ndarray:
    """
    Read a beam position table with columns: beamno, ra, dec
    RA/Dec may be sexagesimal strings ("hh:mm:ss.s", "+dd:mm:ss.s")
    or decimal degrees -- both are handled at match time via SkyCoord.
    """
    return np.loadtxt(
        path, comments="#", dtype=BEAM_DTYPE, 
        #skip_header=1 if _has_text_header(path) else 0,
    )


def _has_text_header(path: str) -> bool:
    """Detect a non-'#' text header line (e.g. 'beamno ra dec')."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            first_tok = line.split()[0]
            try:
                int(first_tok)
                return False
            except ValueError:
                return True
    return False


def attach_beam_coords(cands: np.ndarray, beams: np.ndarray) -> np.ndarray:
    """
    Merge candidate array with beam RA/Dec on beamno.
    Returns a new structured array combining candidate fields + ra/dec,
    dropping candidates with no matching beam.
    """
    beam_lookup = {int(b["beamno"]): (b["ra"], b["dec"]) for b in beams}

    keep_mask = np.array([int(c["beamno"]) in beam_lookup for c in cands])
    n_missing = int((~keep_mask).sum())
    if n_missing:
        print(f"[WARN] {n_missing} candidates have no matching beam position "
              f"and will be dropped.", file=sys.stderr)

    cands = cands[keep_mask]

    out_dtype = np.dtype(CAND_DTYPE.descr + [("ra", "U16"), ("dec", "U16")])
    merged = np.empty(cands.shape[0], dtype=out_dtype)
    for field in CAND_DTYPE.names:
        merged[field] = cands[field]

    ra_vals = np.empty(cands.shape[0], dtype="U16")
    dec_vals = np.empty(cands.shape[0], dtype="U16")
    for i, c in enumerate(cands):
        ra_vals[i], dec_vals[i] = beam_lookup[int(c["beamno"])]
    merged["ra"] = ra_vals
    merged["dec"] = dec_vals

    return merged


def load_local_catalog(path: str, npy: int) -> np.ndarray:
    """Local CSV with columns: name, ra, dec, dm (RA/Dec sexagesimal or deg)."""
    if(not npy):
        return np.loadtxt(
            path, delimiter=",",  dtype=CATALOG_DTYPE,)
    else:
        print("Loading npy")
        return np.load(
            path)
    


def make_skycoord(ra_arr, dec_arr) -> SkyCoord:
    """
    Build a SkyCoord array, auto-detecting sexagesimal vs decimal-degree
    input based on whether the first value contains ':' or a space.
    """
    ra0 = str(ra_arr[0])
    if ":" in ra0 or " " in ra0.strip():
        return SkyCoord(ra=ra_arr, dec=dec_arr,
                         unit=(u.hourangle, u.deg), frame="icrs")
    else:
        return SkyCoord(ra=ra_arr.astype(float), dec=dec_arr.astype(float),
                         unit=(u.deg, u.deg), frame="icrs")


def crossmatch(
    cands: np.ndarray,
    catalog: np.ndarray,
    radius_deg: float = 0.5,
    dm_tol_frac: float = 0.1,
    dm_tol_abs: float = None,
) -> np.ndarray:
    """
    For each candidate, find the nearest known pulsar and flag it as a
    match if separation <= radius_deg AND DM matches within tolerance.
    Returns a structured array: candidate fields + MATCH_EXTRA_DTYPE.
    """
    cand_coords = make_skycoord(cands["ra"], cands["dec"])
    cat_coords = make_skycoord(catalog["ra"], catalog["dec"])

    idx, sep2d, _ = cand_coords.match_to_catalog_sky(cat_coords)

    out_dtype       =   np.dtype(cands.dtype.descr + MATCH_EXTRA_DTYPE)
    out             =   np.empty(cands.shape[0], dtype=out_dtype)
    out_unq         =   np.empty(cands.shape[0], dtype=out_dtype)


    for field in cands.dtype.names:
        out[field] = cands[field]
    
    out["sep_deg"]  = sep2d.deg 
    ddm             = out["dm"].astype(float) - catalog["dm"][idx].astype(float)
    out["ddm"]      = ddm
    out["ddm_frac"] = np.abs(ddm) / catalog["dm"][idx].astype(float)

    pos_ok = out["sep_deg"] <= radius_deg

    if dm_tol_abs is not None:
        dm_ok = np.abs(out["ddm"]) <= dm_tol_abs
    else:
        dm_ok = out["ddm_frac"] <= dm_tol_frac


    mask = pos_ok & dm_ok  # True -> real match, False -> new candidate

    out["match_name"] = np.where(mask, catalog["name"][idx], "NEW_CAND")
    out["match_ra"]   = np.where(mask, catalog["ra"][idx],  cands["ra"])
    out["match_dec"]  = np.where(mask, catalog["dec"][idx], cands["dec"])
    out["match_dm"]   = np.where(mask, catalog["dm"][idx],  out["dm"])
    out["sep_deg"]    = sep2d.deg  

    out["is_match"] = mask
       
    return out


def save_csv(path: str, result: np.ndarray):
    header = ",".join(result.dtype.names)
    header = "#"+header
    fmt = []
    for name in result.dtype.names:
        kind = result.dtype[name].kind
        if kind in ("U", "S"):
            fmt.append("%s")
        elif kind == "b":
            fmt.append("%s")
        elif kind == "i":
            fmt.append("%d")
        else:
            fmt.append("%.6f")
    np.savetxt(path, result, fmt=",".join(fmt), header=header, comments="")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", 
                    required=True,
                    help="Path to candidate file (S/N, sampno, secs, boxcar, idt, dm, beamno, mjd)")
    p.add_argument("--candidates-have-radec", 
                    action="store_true",
                    help="Set if candidate file already contains ra/dec columns "
                        "instead of beamno")
    p.add_argument("--beams", default=None,
                   help="Path to beam position file (beamno, ra, dec). "
                        "Required unless --candidates-have-radec")
    p.add_argument("--catalog", default=None,
                   help="Path to local known-pulsar catalog CSV (name, ra, dec, dm)")
    p.add_argument("--use-atnf", action="store_true",
                   help="Fetch known-pulsar catalog live from ATNF via psrqpy")
    p.add_argument("--radius", type=float, default=0.5,
                   help="Positional match radius in degrees (default: 0.5)")
    p.add_argument("--dm-tol", type=float, default=0.1, dest="dm_tol",
                   help="Fractional DM tolerance, e.g. 0.1 = 10%% (default: 0.1)")
    p.add_argument("--dm-abs-tol", type=float, default=None, dest="dm_abs_tol",
                   help="Absolute DM tolerance in % (overrides --dm-tol if set)")
    p.add_argument("--out", 
                    default="matches.csv",
                    help="Output CSV path for matches (default: matches.csv)")
    p.add_argument("--out_unq", 
                    default="unique.csv",
                    help="Output CSV path for unique sources (default: unique.csv)")
    p.add_argument("--all", action="store_true",
                   help="Write all candidates with nearest-match info, "
                        "not just confirmed matches")
    p.add_argument("--npy_file", default=1,
                   help="Use npy files? Default=1")
 

    return p.parse_args()


def main():
    args = parse_args()

    # --- candidates ---
    if args.candidates_have_radec:
        cands = load_candidates_with_radec(args.candidates)
    else:
        if not args.beams:
            sys.exit("--beams is required unless --candidates-have-radec is set")
        cands = load_candidates(args.candidates)
        beams = read_beam_positions(args.beams)
        cands = attach_beam_coords(cands, beams)

    print(f"[INFO] Loaded {len(cands)} candidates")

    # --- catalog ---
    if args.catalog:
        catalog = load_local_catalog(args.catalog, args.npy_file)
    else:
        sys.exit("Provide --catalog <file>")

    print(f"[INFO] Catalog has {len(catalog)} known pulsars")

    # --- cross-match ---
    result = crossmatch(
        cands, 
        catalog,
        radius_deg=args.radius,
        dm_tol_frac=args.dm_tol,
        dm_tol_abs=args.dm_abs_tol,
    )

    n_match = int(result["is_match"].sum())
    print(f"[INFO] {n_match} / {len(result)} candidates matched a known pulsar "
          f"(radius<={args.radius} deg, dm_tol={args.dm_abs_tol or args.dm_tol})")

    out = result if args.all else result[result["is_match"]]
    unq = result if args.all else result[np.where(result["is_match"]==False)[0]]
    save_csv(args.out, out)
    save_csv(args.out_unq, unq)
    print(f"[INFO] Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
