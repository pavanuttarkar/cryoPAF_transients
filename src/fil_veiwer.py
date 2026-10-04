#!/usr/bin/env python3
"""Simple filterbank viewer.

Usage:
    python fil_viewer.py file.fil -S 10 -T 2 -d 56.7
      -S  seek (seconds from start of file)
      -T  total time to read (seconds)
      -d  dispersion measure to dedisperse at (pc cm^-3)
"""
import argparse
import struct

import matplotlib.pyplot as plt
import numpy as np

INT = {"nchans", "nifs", "nbits", "nsamples", "machine_id", "telescope_id",
       "data_type", "barycentric", "pulsarcentric", "nbeams", "ibeam"}
DBL = {"tstart", "tsamp", "fch1", "foff", "refdm", "az_start", "za_start",
       "src_raj", "src_dej", "period"}
STR = {"source_name", "rawdatafile"}

K_DM = 4.148808e3  # MHz^2 pc^-1 cm^3 s


def read_fil(path, nsamples=None, start=0):
    """Read `nsamples` time samples beginning at sample `start` (float32)."""
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

        f.seek(start * sample_bytes, 1)
        count = -1 if nsamples is None else nsamples * nifs * nchans
        data = np.fromfile(f, dtype=dtype, count=count)

    data = data.reshape(-1, nifs, nchans).astype(np.float32, copy=False)
    return hdr, data


def get_freqs(hdr):
    """Channel frequencies in MHz."""
    return hdr["fch1"] + hdr["foff"] * np.arange(hdr["nchans"])


def dm_shifts(freqs, dm, tsamp):
    """Per-channel delay in samples, relative to the highest frequency."""
    delay = K_DM * dm * (freqs ** -2.0 - freqs.max() ** -2.0)  # seconds
    return np.round(delay / tsamp).astype(int)


def dedisperse(data, shifts, nout):
    """data: (time, chan). Align channels, return (nout, chan)."""
    out = np.empty((nout, data.shape[1]), dtype=np.float32)
    for c, s in enumerate(shifts):
        out[:, c] = data[s:s + nout, c]
    return out


def main():
    p = argparse.ArgumentParser(description="Simple filterbank viewer")
    p.add_argument("file")
    p.add_argument("-S", "--seek", type=float, default=0.0,
                   help="seek into file, seconds (default 0)")
    p.add_argument("-T", "--total", type=float, default=1.0,
                   help="total time to read, seconds (default 1)")
    p.add_argument("-d", "--dm", type=float, default=0.0,
                   help="dedispersion DM in pc cm^-3 (default 0)")
    args = p.parse_args()

    # Read header only to get tsamp, then read the real slice
    hdr, _ = read_fil(args.file, nsamples=0)
    tsamp = hdr["tsamp"]
    freqs = get_freqs(hdr)

    start = int(round(args.seek / tsamp))
    nout = int(round(args.total / tsamp))
    shifts = dm_shifts(freqs, args.dm, tsamp)
    maxshift = int(shifts.max())

    # Read extra samples so the dispersion sweep isn't cut off at the end
    hdr, data = read_fil(args.file, nsamples=nout + maxshift, start=start)
    data = data[:, 0, :]  # total intensity (time, chan)

    nout = min(nout, data.shape[0] - maxshift)
    if nout <= 0:
        raise SystemExit("Not enough data for this seek/total/DM combination.")

    dd = dedisperse(data, shifts, nout) if args.dm else data[:nout]

    # Display: remove per-channel median
    img = dd - np.median(dd, axis=0)
    vmin, vmax = np.percentile(img, [1, 99])
    series = img.sum(axis=1)

    t0 = start * tsamp
    t1 = t0 + nout * tsamp
    t = t0 + tsamp * np.arange(nout)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05})
    ax0.plot(t, series, lw=0.6)
    ax0.set_ylabel("Sum")
    ax0.set_title(f"{hdr.get('source_name', args.file)}   DM={args.dm}   "
                  f"tsamp={tsamp * 1e6:.1f} us")

    flip = hdr["foff"] > 0
    ax1.imshow(img.T, aspect="auto", origin="lower" if flip else "upper",
               extent=[t0, t1, freqs.min(), freqs.max()],
               vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (MHz)")
    plt.show()


if __name__ == "__main__":
    main()
