#!/usr/bin/env python3
"""
fetch_atnf_catalog.py
======================

Pull the ATNF pulsar catalogue (name, RA, Dec, DM) via psrqpy and save it
as a plain CSV file, using only numpy (no pandas).

Usage:
    python fetch_atnf_catalog.py --out known_pulsars.csv
"""

__author__      =   "Pavan Uttarkar and Claude"
__version__     =   "0.0.1"
__maintainer__  =   "Pavan Uttarkar"
__email__       =   "pavan.uttarkar@gmail.com"
__status__      =   "Development"
__project__     =   "cryoPAF transient searches"

import argparse
import numpy as np
from psrqpy import QueryATNF


# Structured dtype for the catalogue. Field widths (U32/U16) are generous
# upper bounds for pulsar JNAMEs and sexagesimal RAJ/DECJ strings; widen
# if psrqpy ever throws truncation warnings.
CATALOG_DTYPE = np.dtype([
    ("name", "U32"),
    ("ra",   "U16"),
    ("dec",  "U16"),
    ("dm",   "f8"),
])


def fetch_catalog(require_dm: bool = True) -> np.ndarray:
    """
    Query ATNF for JNAME, RAJ, DECJ, DM and pack the result into a numpy
    structured array with dtype CATALOG_DTYPE.

    require_dm=True drops pulsars with no DM listed (NaN in the catalogue).
    Filtering is done locally with numpy rather than via psrqpy's
    `condition` kwarg, which expects a full logical expression string
    (e.g. "DM>0"), not just a field name, and was erroring on "dm".
    """
    query = QueryATNF(params=["JNAME", "RAJ", "DECJ", "DM"])
    table = query.table  # astropy Table

    names = np.array(table["JNAME"], dtype=str)
    ra = np.array(table["RAJ"], dtype=str)
    dec = np.array(table["DECJ"], dtype=str)
    dm = np.array(table["DM"], dtype=float)

    catalog = np.empty(names.shape[0], dtype=CATALOG_DTYPE)
    catalog["name"] = names
    catalog["ra"] = ra
    catalog["dec"] = dec
    catalog["dm"] = dm

    if require_dm:
        catalog = catalog[~np.isnan(catalog["dm"])]

    return np.atleast_1d(catalog)


def save_csv(path: str, catalog: np.ndarray):
    """Save a CATALOG_DTYPE structured array as CSV, numpy-only."""
    np.savetxt(
        path,
        catalog,
        fmt="%s,%s,%s,%.6f",
        header="name,ra,dec,dm",
        comments="",
    )


def save_npy(path: str, catalog: np.ndarray):
    """Save the structured array in numpy's native binary format."""
    np.save(path, catalog)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="known_pulsars.csv",
                         help="Output CSV path (default: known_pulsars.csv)")
    parser.add_argument("--npy-out", default=None,
                         help="Optional path to also save as .npy "
                              "(structured array, dtype preserved)")
    parser.add_argument("--no-dm-filter", action="store_true",
                         help="Include pulsars with no DM listed (default: "
                              "excluded)")
    args = parser.parse_args()

    catalog = fetch_catalog(require_dm=not args.no_dm_filter)

    print(f"[INFO] Retrieved {len(catalog)} pulsars from ATNF")
    save_csv(args.out, catalog)
    print(f"[INFO] Saved CSV to {args.out}")

    if args.npy_out:
        save_npy(args.npy_out, catalog)
        print(f"[INFO] Saved structured array (.npy) to {args.npy_out}")


if __name__ == "__main__":
    main()
