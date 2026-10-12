import struct
import numpy as np
from header import *
import logging
import time
import warnings
from numba import njit, prange
 
logger = logging.getLogger(__name__)

# Dispersion constant: delay[s] = K_DM * DM * (f_lo^-2 - f_hi^-2), f in MHz
K_DM = 4.148808e3  # MHz^2 pc^-1 cm^3 s
 
_DM_BLOCK = 8
_T_TILE = 1024



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


@njit(
    parallel=True,
    fastmath={"reassoc", "contract"},
    cache=True,
    nogil=True,
    boundscheck=False,
)
def dedisperse_kernel(data, delays, out):
    """Optimized cache-blocked incoherent dedispersion."""
    ndm, nchans = delays.shape
    nsamples = data.shape[1]
    nout = out.shape[1]
    nblk = (ndm + _DM_BLOCK - 1) // _DM_BLOCK

    # Pre-initialize output array in parallel for maximum memory bandwidth
    for d in prange(ndm):
        out[d, :nout] = 0.0

    for blk in prange(nblk):
        d0 = blk * _DM_BLOCK
        d1 = min(d0 + _DM_BLOCK, ndm)
        
        # Local buffer accumulation per DM block / tile
        for t0 in range(0, nout, _T_TILE):
            t1 = min(t0 + _T_TILE, nout)
            for c in range(nchans):
                row = data[c]
                for d in range(d0, d1):
                    s = delays[d, c]
                    tmax = min(t1, nsamples - s)
                    for t in range(t0, tmax):
                        out[d, t] += row[t + s]


@njit(
    parallel=True,
    fastmath={"reassoc", "contract"},
    cache=True,
    nogil=True,
    boundscheck=False,
)
def dedisperse_kernel_simple(data, delays, out):
    ndm, nchans = delays.shape
    nsamples = data.shape[1]
    nout = out.shape[1]
    for d in prange(ndm):
        acc = out[d]
        acc[:] = 0.0
        for c in range(nchans):
            s = delays[d, c]
            row = data[c]
            tmax = min(nout, nsamples - s)
            for t in range(tmax):
                acc[t] += row[t + s]


@njit(
    parallel=True,
    fastmath={"reassoc", "contract"},
    cache=True,
    nogil=True,
    boundscheck=False,
)
def dedisperse_kernel_partial(data, delays, lo, hi, out):
    ndm, nchans = delays.shape
    nsamples = data.shape[1]
    nout = out.shape[1]
    nblk = (ndm + _DM_BLOCK - 1) // _DM_BLOCK

    for d in prange(ndm):
        out[d, :nout] = 0.0

    for blk in prange(nblk):
        d0 = blk * _DM_BLOCK
        d1 = min(d0 + _DM_BLOCK, ndm)
        
        for t0 in range(0, nout, _T_TILE):
            t1 = min(t0 + _T_TILE, nout)
            for c in range(nchans):
                row = data[c]
                for d in range(d0, d1):
                    s = delays[d, c]
                    tmax = min(t1, nsamples - s)
                    for t in range(t0, tmax):
                        out[d, t] += row[t + s]

        # Optimized difference array generation for partial band count
        diff = np.zeros((d1 - d0, nout + 1), dtype=np.int32)
        for c in range(nchans):
            lo_c = lo[c]
            hi_c = hi[c]
            for d in range(d0, d1):
                s = delays[d, c]
                a = max(lo_c - s, 0)
                b = min(hi_c - s, nout)
                if a < b:
                    diff[d - d0, a] += 1
                    diff[d - d0, b] -= 1

        for d in range(d0, d1):
            n = 0
            row_out = out[d]
            row_diff = diff[d - d0]
            factor = np.float32(nchans)
            for t in range(nout):
                n += row_diff[t]
                if n > 0:
                    row_out[t] = row_out[t] * np.sqrt(factor / n)
                else:
                    row_out[t] = np.nan


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def channel_freqs(hdr):
    return hdr["fch1"] + hdr["foff"] * np.arange(hdr["nchans"])


def delay_table(hdr, dms):
    f = channel_freqs(hdr)
    f_ref = f.max()
    dt = K_DM * (f ** -2.0 - f_ref ** -2.0)
    delays = np.rint(np.outer(np.asarray(dms), dt) / hdr["tsamp"]).astype(np.int64)
    logger.debug("Delay table: %d DMs x %d chans, f_ref=%.3f MHz, max delay=%d samples",
                 delays.shape[0], delays.shape[1], f_ref, delays.max())
    return delays


def max_delay_samples(hdr, dm):
    return int(delay_table(hdr, [dm]).max())


def _run_kernel(data, delays, out, skip_nan):
    if not skip_nan:
        dedisperse_kernel(data, delays, out)
        return
    nsamples = data.shape[1]
    finite = np.isfinite(data)
    anyf = finite.any(axis=1)
    lo = np.where(anyf, finite.argmax(axis=1), 0).astype(np.int64)
    hi = np.where(anyf, nsamples - finite[:, ::-1].argmax(axis=1), 0).astype(np.int64)
    if not finite.all():
        data = np.where(finite, data, 0.0).astype(np.float32)
    dedisperse_kernel_partial(data, delays, lo, hi, out)


def central_dm(dm_min, dm_max):
    return 0.5 * (dm_min + dm_max)


def _read_header(path):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        hdr, _ = read_fil(path, nsamples=0)
    logger.debug("Header of %s: nchans=%s fch1=%s foff=%s tsamp=%s nbits=%s",
                 path, hdr.get("nchans"), hdr.get("fch1"), hdr.get("foff"),
                 hdr.get("tsamp"), hdr.get("nbits"))
    return hdr


# ---------------------------------------------------------------------------
# 2. DM transform
# ---------------------------------------------------------------------------
def dm_transform(data, hdr, dm_min, dm_max, dm_step, skip_nan=False):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (nchans, nsamples), got shape {data.shape}")
    if dm_step <= 0 or dm_max < dm_min:
        raise ValueError(f"Invalid DM range: min={dm_min}, max={dm_max}, step={dm_step}")

    dms = np.arange(dm_min, dm_max + 0.5 * dm_step, dm_step)
    delays = delay_table(hdr, dms)
    dm_c = central_dm(dm_min, dm_max)
    max_delay = max_delay_samples(hdr, dm_c)

    nchans, nsamples = data.shape
    if len(dms) == 1:
        logger.warning("Only ONE DM trial (dm_min=%s, dm_max=%s, dm_step=%s)", dm_min, dm_max, dm_step)
    logger.info("DM transform: DM %.3f-%.3f step %.3f (%d trials), input %d chans x %d samples",
                dms[0], dms[-1], dm_step, len(dms), nchans, nsamples)

    if nsamples <= max_delay:
        logger.error("Not enough samples: need > %d for central DM %.3f, got %d", max_delay, dm_c, nsamples)
        raise ValueError(f"Need more than {max_delay} samples for central DM {dm_c}, got {nsamples}")
    nout = nsamples - max_delay
    logger.info("Central DM %.3f: max delay = %d samples (%.3f s); output %d samples/trial",
                dm_c, max_delay, max_delay * hdr["tsamp"], nout)

    over = delays.max(axis=1) - max_delay
    n_incomplete = int((over > 0).sum())
    if n_incomplete:
        logger.warning("%d/%d DM trials need up to %d more samples", n_incomplete, len(dms), int(over.max()))

    data = np.ascontiguousarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    dead = ~finite.any(axis=1)
    if dead.any():
        logger.warning("%d/%d channels are entirely NaN/inf", dead.sum(), nchans)
    elif not finite.all():
        logger.info("Input contains %d NaN/inf samples", int((~finite).sum()))

    out = np.empty((len(dms), nout), dtype=np.float32)
    logger.debug("Output buffer: %.1f MB", out.nbytes / 1e6)

    t0 = time.perf_counter()
    _run_kernel(data, delays, out, skip_nan)
    dt = time.perf_counter() - t0
    gflop = len(dms) * nchans * nout / 1e9
    logger.info("Kernel finished in %.3f s (%.2f Gadd/s)", dt, gflop / dt)
    return out.T, dms


def dedisperse_dm(data, hdr, dm, nout=None, skip_nan=False):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (nchans, nsamples), got shape {data.shape}")
    nchans, nsamples = data.shape
    delays = delay_table(hdr, [dm])
    max_delay = int(delays.max())
    if nsamples <= max_delay:
        logger.error("Not enough samples: need > %d for DM %.3f, got %d", max_delay, dm, nsamples)
        raise ValueError(f"Need more than {max_delay} samples for DM {dm}, got {nsamples}")
    complete = nsamples - max_delay
    if nout is None:
        nout = complete
    elif nout > complete:
        logger.warning("nout=%d > %d complete samples at DM %.3f", nout, complete, dm)
    nout = min(nout, nsamples)
    data = np.ascontiguousarray(data, dtype=np.float32)
    out = np.empty((1, nout), dtype=np.float32)
    _run_kernel(data, delays, out, skip_nan)
    return out[0]


# ---------------------------------------------------------------------------
# 3. Compute delay and read the required samples
# ---------------------------------------------------------------------------
def read_for_dm(path, start, window, dm_min, dm_max, pad=False):
    hdr = _read_header(path)
    dm_c = central_dm(dm_min, dm_max)
    max_delay = max_delay_samples(hdr, dm_c)
    nsamples = window + max_delay
    front = max(0, -start)
    first = max(start, 0)
    need = nsamples - front
    if need <= 0:
        raise ValueError(f"The whole read lies before the start of the file")
    
    t0 = time.perf_counter()
    hdr, got = read_fil(path, nsamples=max(need, 0), start=first)
    logger.info("Read %s in %.3f s", got.shape, time.perf_counter() - t0)
    
    back = need - got.shape[1]
    if back > 0 and not pad:
        raise ValueError(f"Not enough samples from start={start}")
    if front or back > 0:
        data = np.full((got.shape[0], nsamples), np.nan, dtype=np.float32)
        data[:, front:front + got.shape[1]] = got
    else:
        data = got
    return hdr, data, max_delay


def start_for_candidate(hdr, cand_sample, dm_cand, window, f_ref=None, pad=False):
    f_top = channel_freqs(hdr).max()
    if f_ref is None:
        f_ref = f_top
    delta_s = K_DM * dm_cand * (f_ref ** -2.0 - f_top ** -2.0)
    shift = delta_s / hdr["tsamp"]
    sample_top = cand_sample - shift
    start = int(round(sample_top)) - window // 2
    if sample_top < 0:
        arrival = sample_top + delay_table(hdr, [dm_cand])[0]
        n_in = int((arrival >= 0).sum())
        nchans = len(arrival)
        if not pad or n_in == 0:
            raise ValueError(f"Candidate's arrival at top of band is before file start.")
    if start < 0 and not pad:
        raise ValueError(f"Candidate is too close to file start.")
    return start
