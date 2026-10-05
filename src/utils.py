import struct
import numpy as np
from header import *


INT = {"nchans", "nifs", "nbits", "nsamples", "machine_id", "telescope_id",
       "data_type", "barycentric", "pulsarcentric", "nbeams", "ibeam"}
DBL = {"tstart", "tsamp", "fch1", "foff", "refdm", "az_start", "za_start",
       "src_raj", "src_dej", "period"}
STR = {"source_name", "rawdatafile"}


def dump_npy(file, save_file) -> None:
    data    =   np.loadtxt(f"{file}", 
				dtype=CATALOG_DTYPE, 
				delimiter=',')
    np.save(f"{save_file}", data)
    return None

def read_fil(path, nsamples=None, start=0):
    """Read `nsamples` time samples beginning at sample `start`.
    nsamples=None reads to the end of the file."""
    hdr = {}
    with open(path, "rb") as f:
        def read_str():
            n = struct.unpack("<i", f.read(4))[0]
            return f.read(n).decode()

        assert read_str() == "HEADER_START", "Not a filterbank file"
        while (key := read_str()) != "HEADER_END":
            if key in INT:
                hdr[key] = struct.unpack("<i", f.read(4))[0]
            elif key in DBL:
                hdr[key] = struct.unpack("<d", f.read(8))[0]
            elif key in STR:
                hdr[key] = read_str()
            else:
                raise ValueError(f"Unknown header key: {key}")

        dtype = np.dtype({8: np.uint8, 16: np.uint16, 32: np.float32}[hdr["nbits"]])
        nifs, nchans = hdr["nifs"], hdr["nchans"]
        sample_bytes = nifs * nchans * dtype.itemsize

        f.seek(start * sample_bytes, 1)          # skip `start` samples
        count = -1 if nsamples is None else nsamples * nifs * nchans
        data = np.fromfile(f, dtype=dtype, count=count)

    data = data.reshape(-1, nifs, nchans)        # (time, ifs, freq)
    data    =   data[:,0].T 
    data    =   (data - np.nanmean(data, axis=1)[:,None])/np.std(data, axis=1)[:,None]
    return hdr, data

def get_freqs(hdr):
    return hdr["fch1"] + hdr["foff"] * np.arange(hdr["nchans"])

def _excess_kurtosis(x, axis=-1):
    mu = np.nanmean(x, axis=axis, keepdims=True)
    d = x - mu
    m2 = np.nanmean(d**2, axis=axis)
    m4 = np.nanmean(d**4, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return m4 / m2**2 - 3.0


def _mad_sigma(x):
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

def kurtosis_excise(data, nsigma=4.0, fill="nan", inplace=False):
    """
    Fast whole-channel RFI excision using excess kurtosis.

    Parameters
    ----------
    data : ndarray (nchan, ntime), as returned by read_fil.
    nsigma : threshold in robust (MAD) sigmas from the median channel kurtosis.
    fill : "noise" (N(0,1), default), "zero" or "nan".
    inplace : modify `data` directly (avoids a full copy).
    seed : RNG seed for the noise fill.

    Returns
    -------
    clean : ndarray (nchan, ntime)
    bad_chan : bool ndarray (nchan,), True for excised channels.
    """
    n = data.shape[1]

    # FMA is faster, so using einsym - Raw moments via einsum: single pass each, no large temporaries
    # beyond x^2.
    x2 = data * data
    mu = data.sum(axis=1) / n
    m2r = np.einsum("ij,ij->i", data, data) / n        
    m3r = np.einsum("ij,ij->i", x2, data) / n          
    m4r = np.einsum("ij,ij->i", x2, x2) / n            
    del x2

    # Convert to central moments
    m2 = m2r - mu**2
    m4 = m4r - 4 * mu * m3r + 6 * mu**2 * m2r - 3 * mu**4

    with np.errstate(divide="ignore", invalid="ignore"):
        k = m4 / m2**2 - 3.0

    # Robust thresholding across channels
    good        = np.isfinite(k)
    med         = np.median(k[good])
    sigma       = 1.4826 * np.median(np.abs(k[good] - med))
    bad_chan    = ~good | (np.abs(k - med) > nsigma * sigma)

    # Fill only the flagged channels
    clean       = data if inplace else data.copy()
    idx         = np.flatnonzero(bad_chan)
    if idx.size:
        if fill == "zero":
            clean[idx] = 0.0
        elif fill == "nan":
            clean[idx] = np.nan
        else:
            raise ValueError("fill must be 'noise', 'zero' or 'nan'")

    return clean, bad_chan
