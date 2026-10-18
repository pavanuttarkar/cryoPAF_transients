import os
import sys
import time
import struct
import logging
import warnings
import numpy as np
from header import *

from numba import njit, prange
from matplotlib.pyplot import *


DM_TOL = 1.0      # pc/cc, max DM difference to call two candidates the same
SEC_TOL = 0.01    # s, max time difference to call two candidates the same

OUT_DTYPE       = np.dtype(CAND_DTYPE_CLS.descr + [("score", "i1")])
TELESCOPE       = "Murriyang cryoPAF"

 
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
    nsamples    = data.shape[1]
    finite      = np.isfinite(data)
    anyf        = finite.any(axis=1)
    lo          = np.where(anyf, finite.argmax(axis=1), 0).astype(np.int64)
    hi          = np.where(anyf, nsamples - finite[:, ::-1].argmax(axis=1), 0).astype(np.int64)
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
    #print(data.shape, out.shape)
    
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


_ROWS = {
     5: (-0.24, [67, 68, 69]),
     4: (-0.32, [52, 53, 54, 55, 56]),
     3: (-0.40, [51, 30, 31, 32, 33, 57]),
     2: (-0.48, [50, 29, 14, 15, 16, 34, 58, 70]),
     1: (-0.56, [49, 28, 13, 4, 5, 17, 35, 59, 71]),
     0: (-0.64, [48, 27, 12, 3, 0, 6, 18, 36, 60]),
    -1: (-0.72, [66, 47, 26, 11, 2, 1, 7, 19, 37]),
    -2: (-0.64, [65, 46, 25, 10, 9, 8, 20, 38]),
    -3: (-0.56, [64, 45, 24, 23, 22, 21, 39]),
    -4: (-0.32, [44, 43, 42, 41, 40]),
    -5: (-0.08, [63, 62, 61]),
}

BEAM_SPACING = 0.16                           # deg, centre-to-centre
BEAM_RADIUS = BEAM_SPACING / 2                # deg
ROW_SPACING = BEAM_SPACING * np.sqrt(3) / 2   # deg


def beam_positions():
    """Return {beam_id: (cross_el, el)} offsets in degrees."""
    pos = {}
    for j, (x0, ids) in _ROWS.items():
        for k, b in enumerate(ids):
            pos[b] = (x0 + k * BEAM_SPACING, j * ROW_SPACING)
    return pos


def plot_beam_footprint(ax, highlight=(), title=None, radius=BEAM_RADIUS,
                        label=True, lim=0.9, highlight_color="gray",
                        highlight_alpha=0.6, fontsize=8):
    """Draw the beam footprint on a subplot axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Subplot axes to draw on (e.g. from fig.add_subplot or plt.subplots).
    highlight : array-like of int
        Beam numbers to fill with gray.
    title : str, optional
        Axes title.
    radius : float
        Beam circle radius in degrees.
    label : bool
        Annotate each beam with its ID.
    lim : float
        Symmetric axis limit in degrees.
    highlight_color : str
        Fill colour for highlighted beams.
    highlight_alpha : float
        Fill opacity for highlighted beams.
    fontsize : int
        Beam label font size.

    Returns
    -------
    ax
    """
    highlight = set(int(b) for b in np.atleast_1d(highlight))

    for b, (x, y) in beam_positions().items():
        if b in highlight:
            circ = Circle((x, y), radius, fc=highlight_color,
                          alpha=highlight_alpha, ec="none")
            ax.add_patch(circ)
        # Outline drawn separately so it stays fully opaque
        ax.add_patch(Circle((x, y), radius, fc="none", ec="k", lw=1.0))
        if label:
            ax.text(x, y, str(b), ha="center", va="center", fontsize=fontsize)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, color="0.8", lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cross-elevation offset (deg)")
    ax.set_ylabel("Elevation offset (deg)")
    if title:
        ax.set_title(title)
    return ax


def get_dispersed(data, hdr, window, nf_out=128, nt_out=1024):
    """
    Dispersed dynamic spectrum on the SAME time axis as the dedispersed panels.

    data   : (nchans, window + max_delay) from read_for_dm. data[:, 0] is sample `start`,
             so the candidate's top-of-band arrival (the bow-tie time) is at window//2.
    window : samples shown, identical to the other panels.

    Returns
    -------
    disp  : (nf, nt) float32, ascending frequency, centre column = bow-tie time
    freqs : (nf,) MHz, ascending
    nt    : number of time columns kept (the time axis is time[0] .. time[nt*tb - 1])
    """
    f = channel_freqs(hdr)
    order = np.argsort(f)
    nchans = len(f)

    disp = np.asarray(data[order, :window], dtype=np.float32)
    freqs = f[order]

    fb = max(1, nchans // nf_out)
    tb = max(1, window // nt_out)
    nfc = nchans // fb * fb
    ntc = window // tb * tb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp = np.nanmean(
            disp[:nfc, :ntc].reshape(nfc // fb, fb, ntc // tb, tb), axis=(1, 3))
    freqs = freqs[:nfc].reshape(-1, fb).mean(axis=1)
    return disp, freqs, ntc



def plot_cands(cluster_file, 
                fil_path, 
                window, 
                time_res,
                #highlight_beams,
                out_dir=None, 
                show=False) -> None:
    """
    Generate TransientX-style candidate plots (e.g. to push to Mattermost).

    cluster_file : (str) clustered candidate file
    fil_path     : (str) filterbank file
    window       : (int) number of samples per plot
    time_res     : (float) sample time in seconds
    out_dir      : (str) if given, save PDFs there
    show         : (bool) show() each candidate
    """
    #cls_cands = np.atleast_1d(np.loadtxt(f"{cluster_file}", dtype=CAND_DTYPE_CLS))
    cls_cands   =   np.atleast_1d(read_beam_cands(f"{cluster_file}", dtype=CAND_DTYPE_CLS))
    
    #Reading the beams from file
    highlight_beams     =       cls_cands['beams'] 

    #Reading the beam file:
    

    hdr         = _read_header(fil_path)
    nchans      = hdr["nchans"]
    bw          = abs(hdr["foff"]) * nchans
    fcen        = hdr["fch1"] + hdr["foff"] * (nchans - 1) / 2.0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for i in range(len(cls_cands)):
        c  = cls_cands[i]
        dm = c["dm"]

        start = start_for_candidate(hdr, c["sampno"], dm, window,
                                    f_ref=hdr["fch1"], pad=True)
        hdr_i, data, max_delay = read_for_dm(fil_path, start, window, 0.0, 2 * dm)
        trials, dms = dm_transform(data, hdr_i, 0.0, 2 * dm, 1.0)
        series      = dedisperse_dm(data, hdr_i, dm, nout=trials.shape[0])

        time     = np.arange(window) * time_res
        tpeak    = time[np.argmax(series)]
        width_ms = c["boxcar"] * time_res * 1e3

        disp, d_f, ntc = get_dispersed(data, hdr_i, window)


        # ---- text columns for the info row ----
        col1 = (#"CANDIDATE\n---------\n"
                f"fredda S/N    : {c['sn']:.2f}\n"
                f"DM (pc cm^-3) : {dm:.3f}\n"
                f"Time (s)      : {c['secs']:.4f}\n"
                f"Sample no.    : {int(c['sampno'])}\n"
                f"Peak time     : {tpeak:.4f} s\n"#)
        #col2 = ("\n\n"
                f"Boxcar        : {int(c['boxcar'])} samp\n"
                f"Width         : {width_ms:.2f} ms\n"
                f"IDT           : {int(c['idt'])}\n"
                f"Beam          : {int(c['beamno'])}\n"
                f"Class         : {int(c['cls'])}\n"
                f"N cands       : {int(c['ncands'])}\n")
        col3 = (#"HEADER\n------\n"
                f"Source        : {hdr['source_name']}\n"
                f"Telescope     : {TELESCOPE}\n"
                f"Machine       : {hdr['machine_id']}\n"
                f"Tstart MJD    : {hdr['tstart']:.6f}\n"
                f"Cand MJD      : {c['mjd']:.8f}\n"#)
        #col4 = ("\n\n"
                f"Tsamp         : {hdr['tsamp']*1e3:.2f} ms\n"
                f"Nchans        : {nchans}\n"
                f"fch1/foff     : {hdr['fch1']:.1f}/{hdr['foff']:.1f} MHz\n"
                f"BW / fcen     : {bw:.0f} / {fcen:.0f} MHz\n"
                f"Nbits/Nifs    : {hdr['nbits']}/{hdr['nifs']}\n"
                f"File          : {os.path.basename(fil_path)}\n")


        fig, ax = subplots(
                        nrows=4, 
                        ncols=1,
                        figsize=(10, 14),
                        gridspec_kw={"height_ratios": [1.0, 0.7, 1., 1.0], "hspace": 0}
                        )

        # row 0: info text (left) + beam footprint inset (right)
        ax[0].axis("off")
        for x, y, txt in [(0.00, 1.00, col1), 
                          (0.30, 1.00, col3),
                          #(0.00, 0.50, col3), 
                          #(0.30, 0.50, col3)
                         ]:
    
            ax[0].text(x, y, 
                        txt, 
                        va="top", 
                        ha="left", 
                        family="monospace",
                        fontsize=8, 
                        transform=ax[0].transAxes)
        

        ax_beam = ax[0].inset_axes([0.65, 0.42, 0.35, 0.60])
        plot_beam_footprint(ax_beam, 
                            highlight=highlight_beams, 
                            title="FRB beams")
   



        ax[3].imshow(disp, aspect="auto", origin="lower", cmap="viridis",
                     extent=[time[0], time[ntc - 1], d_f[0], d_f[-1]],
                     interpolation="nearest",
                     vmin=np.nanpercentile(disp, 1),
                     vmax=np.nanpercentile(disp, 99.5))
        ax[3].axvline(tpeak, color="tab:red", ls="--", lw=0.8, alpha=0.6)
        ax[3].set_xlim(time[0], time[-1])
        ax[3].set_xlabel("Time (s)")
        ax[3].set_ylabel("Frequency (MHz)")
        ax[2].sharex(ax[1])      # was ax[2].sharex(ax[2])
        ax[3].sharex(ax[1])
        ax[2].tick_params(labelbottom=False)
 
        # row 2: dedispersed pulse, row 3: DM-time (touching, shared x)
        #ax[2].sharex(ax[2])
        ax[1].plot(time, series, color="k", lw=1.2)
        ax[1].axvline(tpeak, color="tab:red", ls="--", lw=0.8, alpha=0.6)
        ax[1].set_xlim(time[0], time[-1])
        ax[1].set_ylabel("Amplitude")
        ax[1].tick_params(labelbottom=False)

        ax[2].imshow(trials.T, aspect="auto", origin="lower", cmap="viridis",
                     extent=[time[0], time[-1], dms[0], dms[-1]],
                     interpolation="nearest")
        ax[2].axhline(dm, color="w", ls=":", lw=0.6, alpha=0.5)
        ax[2].set_ylabel(r"DM (pc cm$^{-3}$)")
        ax[2].set_xlabel("Time (s)")




        fig.align_ylabels([ax[1], ax[2]])
        fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.06)

        if out_dir:
            fig.savefig(os.path.join(
                out_dir, f"cand_{i:04d}_DM{dm:.2f}_SN{c['sn']:.1f}.pdf"))
        if show:
            plt.show()   # not show(): the `show` argument shadows the function
        close(fig)
    return None



def get_score(frac):
    if frac == 1.0:     # in all beams
        return -1
    if frac < 0.5:      # in fewer than 50% of beams
        return 1
    return 0            # in between (e.g. ~70% of beams)


def get_score(frac):
    if frac == 1.0:     # in all beams
        return -1
    if frac < 0.5:      # in fewer than 50% of beams
        return 1
    return 0            # in between (e.g. ~70% of beams)



def beam_coincidence(fname,
                     dm_tol,
                     sec_tol,
                     nbeams=None,
                     snr_field="sn"):
    """
    Cluster candidates in (DM, time) and keep only the highest-S/N candidate
    of each cluster. Each kept candidate gets a coincidence score: the
    percentage of beams that have at least one candidate within
    (dm_tol, sec_tol) of it.
 
    """
    cands = np.loadtxt(fname, dtype=CAND_DTYPE_CLS, ndmin=1)   # ndmin=1: single-row files
    n = len(cands)
    if nbeams is None:
        nbeams = len(np.unique(cands["beamno"]))
 
    out      = np.zeros(n, dtype=OUT_DTYPE)
    out_beam = np.zeros(n, dtype=CAND_DTYPE_CLS_BEAM)
    for name in CAND_DTYPE_CLS.names:
        out[name]      = cands[name]
        out_beam[name] = cands[name]
 
    # Brightest first; ties broken deterministically (independent of input order)
    order = np.lexsort((cands["beamno"], cands["dm"], cands["secs"], -cands[snr_field]))
 
    claimed = np.zeros(n, dtype=bool)
    keep = []
    for i in order:
        if claimed[i]:
            continue
 
        c = cands[i]
        near = (np.abs(cands["dm"]   - c["dm"])   <= dm_tol) & \
               (np.abs(cands["secs"] - c["secs"]) <= sec_tol)
 
        beams = np.unique(cands["beamno"][near])          
        out["score"][i]      = (len(beams) / nbeams) * 100
        out_beam["beams"][i] = beams
 
        claimed |= near                                   
        keep.append(i)
 
    keep     = np.sort(np.asarray(keep, dtype=np.intp))   
    out      = out[keep]
    out_beam = out_beam[keep]
 
    np.savetxt(f"{fname}.con",
               out,
               fmt=["%.2f", "%d", "%.4f", "%d", "%d", "%.2f", "%d", "%.6f", "%d", "%d", "%.2f"],
               header=",".join(out.dtype.names))
 
    write_beam_cands(f"{fname}.beams_con",
                     out_beam,
                     fmt='.4g',
                     field_fmt={'mjd': '.8f', 'sampno': '.0f', 'beamno': '.0f'},
                     array_fmt='1.0f')
    return out
 
"""
def beam_coincidence(fname, 
                    dm_tol, 
                    sec_tol, 
                    output,
                    nbeams=None):
    cands = np.loadtxt(fname, dtype=CAND_DTYPE_CLS)
    if nbeams is None:
        nbeams = len(np.unique(cands["beamno"]))

    out         = np.zeros(len(cands), dtype=OUT_DTYPE)
    out_beam    = np.zeros(len(cands), dtype=CAND_DTYPE_CLS_BEAM)
    
    for name in CAND_DTYPE_CLS.names:
        out[name]       = cands[name]
        out_beam[name]  = cands[name]


    for i, c in enumerate(cands):
        match = (np.abs(cands["dm"] - c["dm"]) <= dm_tol) & \
                (np.abs(cands["secs"] - c["secs"]) <= sec_tol)
        nmatch = len(np.unique(cands["beamno"][match]))   # distinct beams (includes own beam)
        out["score"][i]                 = (nmatch / nbeams) * 100
        out_beam["beams"][i]   = np.unique(cands["beamno"][match])     

    np.savetxt(f"{fname}.con", 
                out, 
                fmt=["%.2f", "%d", "%.4f", "%d", "%d", "%.2f", "%d", "%.6f", "%d", "%d", "%d"], 
                header=",".join(out.dtype.names))

<<<<<<< HEAD



=======
    write_beam_cands(f"{fname}.beams_con", 
                    out_beam, 
                    fmt='.4g',
                    field_fmt={'mjd': '.8f', 'sampno': '.0f', 'beamno': '.0f'},
                    array_fmt='1.0f'
                    )
    return out
"""

def write_beam_cands(filename, 
                    data, 
                    dtype=CAND_DTYPE_CLS_BEAM,
                    fmt='.6g', 
                    field_fmt=None, 
                    array_fmt=None):

    names       = dtype.names
    field_fmt   = field_fmt or {}        # per-field overrides, e.g. {'mjd': '.8f'}
    array_fmt   = array_fmt or fmt       # format for the variable array values
    header      = '# ' + ' '.join(names[:-1]) + f' {names[-1]}...'
    with open(filename, 'w') as f:
        f.write(header + '\n')
        for r in data:
            scalars     = [format(r[n], field_fmt.get(n, fmt)) for n in names[:-1]]
            arr         = np.asarray(r[names[-1]], dtype=float)
            f.write(' '.join(scalars + [format(v, array_fmt) for v in arr]) + '\n')

def read_beam_cands(filename, dtype=CAND_DTYPE_CLS_BEAM):
    n_scalar = len(dtype.names) - 1
    with open(filename) as f:
        lines = [l for l in f.read().splitlines()
                 if l.strip() and not l.lstrip().startswith('#')]

    out = np.empty(len(lines), dtype=dtype)
    for i, line in enumerate(lines):
        t = line.split()
        scalars = [float(x) for x in t[:n_scalar]]
        arr = np.array(t[n_scalar:], dtype=float)    # rest of the line
        out[i] = (*scalars, arr)
    return out
>>>>>>> be44e8d95555ea427bc6ae685a77e005b060ba83
