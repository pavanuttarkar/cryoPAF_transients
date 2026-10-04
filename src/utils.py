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

