import re
import h5py
import glob
from collections import defaultdict
from typing import Dict
import numpy as np


DATE_PATTERN = re.compile(r"_(\d{8})/")  # matches _20240403/

def sorted_files_by_date(pattern: str) -> list[str]:
    """Glob paths matching pattern, sorted by the 8-digit date segment
    that appears right before a path separator, e.g. '..._20241217/...'"""
    date_re = re.compile(r'_(\d{8})/')

    def get_date(path: str) -> str:
        match = date_re.search(path)
        return match.group(1) if match else ''

    matches = glob.glob(pattern, recursive=True)
    return sorted(matches, key=get_date)



def extract_date(path: str) -> str:
    """Extract the YYYYMMDD date string from a path like .../FRB20240114A_20240403/..."""
    match = DATE_PATTERN.search(path)
    if not match:
        raise ValueError(f"Could not find a date (YYYYMMDD) in path: {path}")
    return match.group(1)


def get_stats(path_glob: str) -> Dict[str, np.ndarray]:
    """
    Build RFI masks from HDF5 files matching `path_glob`, grouped by date.

    Parameters
    ----------
    path_glob : str
        A glob pattern, e.g.
        "../../<path>/<path>/FRB*_*/search_dir/subband_1/raw/*/rfi/*"

    Returns
    -------
    dict[str, np.ndarray]
        Maps date string (YYYYMMDD) -> RFI mask array of shape (n_files_for_date, n_channels).
    """
    n_channels = 3328 * 2
    fils =  sorted_files_by_date(path_glob) 

    if not fils:
        raise FileNotFoundError(f"No files matched pattern: {path_glob}")

    # First pass: group file paths by date
    files_by_date = defaultdict(list)
    for fil in fils:
        files_by_date[extract_date(fil)].append(fil)

    idx=0
    # Second pass: build the mask per date
    masks_by_date = np.zeros((len(files_by_date), 3328*2))
    for date, date_fils in files_by_date.items():
        rfi_mask = np.zeros((len(date_fils), n_channels), dtype=np.uint8)
        for i, fil in enumerate(date_fils):
            with h5py.File(fil, "r") as f1:
                bad_chans = np.asarray(f1["bad_chans"])
                if(len(bad_chans)!=3328*2):
                    bad_chans=np.zeros((3328*2))
                    bad_chans[:] = np.nan
            rfi_mask[i, bad_chans.astype(bool)] = 1

        masks_by_date[idx] = np.nanmean(rfi_mask, axis=0)
        idx=idx+1
    return masks_by_date

if __name__ == '__main__':
    get_stats
