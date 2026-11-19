import argparse
import ipaddress
import math
import socket
import struct
import sys
import numpy as np
from numba import njit
from src.header import *

UNVISITED = -2
NOISE     = -1


def load_fredda(path, dt):
    # columns: S/N, sampno, secs, boxcar, idt, dm, beamno, mjd
    fredda_cands = np.atleast_1d(np.loadtxt(path, dtype=CAND_DTYPE))
    ok   = fredda_cands['idt'] > 0
    disp = dt / np.median(fredda_cands['dm'][ok] / fredda_cands['idt'][ok])   # sweep [s per DM unit]
    return fredda_cands, disp



def parse_packet(data):
    """One datagram (whitespace separated text, 8 values per candidate, same column
    order as the text file) -> structured array of dtype CAND_DTYPE."""
    names = np.dtype(CAND_DTYPE).names
    if isinstance(data, bytes):
        data = data.decode('ascii', errors='ignore')
    data = "\n".join(l for l in data.splitlines() if not l.lstrip().startswith('#'))
    vals = np.fromstring(data, sep=' ')
    if vals.size % len(names) != 0:
        raise ValueError(f"packet has {vals.size} values, not a multiple of {len(names)}")
    vals = vals.reshape(-1, len(names))

    cands = np.empty(vals.shape[0], dtype=CAND_DTYPE)
    for i, name in enumerate(names):
        cands[name] = vals[:, i]
    return cands


def open_socket(host, port):
    """Bind a UDP socket on host:port; join the multicast group if host is one."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
    sock.bind((host, port))
    if ipaddress.ip_address(host).is_multicast:
        mreq = struct.pack('=4sl', socket.inet_aton(host), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    return sock


def read_packets(sock, max_packets=100, timeout=10.0, bufsize=65535):
    """DUMMY reader. Pulls datagrams off an already-open socket and returns all of
    their candidates as one structured array (dtype CAND_DTYPE). Stops after
    max_packets datagrams, or after `timeout` seconds with nothing received.
    Replace with the real block-handling logic."""
    sock.settimeout(timeout)
    chunks = []
    n = 0
    while max_packets <= 0 or n < max_packets:
        n += 1
        try:
            data, addr = sock.recvfrom(bufsize)
        except (socket.timeout, KeyboardInterrupt):
            break
        try:
            chunks.append(parse_packet(data))
        except ValueError as err:
            print(f"skipping bad packet from {addr}: {err}", file=sys.stderr)

    if not chunks:
        return np.empty(0, dtype=CAND_DTYPE)
    return np.concatenate(chunks)


def load_udp(hostport, dt, max_packets=100, timeout=10.0):
    """UDP counterpart of load_fredda: returns (cands, disp)."""
    host, port = hostport.split(':')
    sock = open_socket(host, int(port))
    try:
        fredda_cands = read_packets(sock, max_packets, timeout)
    finally:
        sock.close()
    if len(fredda_cands) == 0:
        raise SystemExit(f"no candidates received from {hostport}")
    ok   = fredda_cands['idt'] > 0
    disp = dt / np.median(fredda_cands['dm'][ok] / fredda_cands['idt'][ok])   # sweep [s per DM unit]
    return fredda_cands, disp


@njit(cache=True)
def _region_query(i, t, y, w, eps, lim, buf):
    """Fill buf with all j where hypot(dt, dy) <= eps * max(w_i, w_j). Self is included.
    t is sorted, so only a time window of +-lim needs to be scanned."""
    n = t.shape[0]
    k = 0
    j = i
    while j >= 0 and t[i] - t[j] <= lim:                    # scan backwards (includes i itself)
        dx = t[i] - t[j]
        dy = y[i] - y[j]
        s  = w[i] if w[i] > w[j] else w[j]
        if math.sqrt(dx * dx + dy * dy) <= eps * s:
            buf[k] = j
            k += 1
        j -= 1
    j = i + 1
    while j < n and t[j] - t[i] <= lim:                     # scan forwards
        dx = t[j] - t[i]
        dy = y[i] - y[j]
        s  = w[i] if w[i] > w[j] else w[j]
        if math.sqrt(dx * dx + dy * dy) <= eps * s:
            buf[k] = j
            k += 1
        j += 1
    return k


@njit(cache=True)
def _dbscan(t, y, w, eps, min_samples):
    """DBSCAN with the pair metric hypot(dt, dy) / max(w_i, w_j). t must be sorted."""
    n      = t.shape[0]
    labels = np.full(n, UNVISITED, dtype=np.int64)
    buf    = np.empty(n, dtype=np.int64)
    stack  = np.empty(n, dtype=np.int64)
    lim    = eps * w.max()                                  # no pair can be further apart in time
    c      = 0

    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        k = _region_query(i, t, y, w, eps, lim, buf)
        if k < min_samples:                                 # self counts, same as sklearn
            labels[i] = NOISE
            continue

        labels[i] = c
        sp = 0
        for m in range(k):
            q = buf[m]
            if labels[q] == UNVISITED:
                stack[sp] = q
                sp += 1
                labels[q] = c
            elif labels[q] == NOISE:                        # border point, joins but is not expanded
                labels[q] = c

        while sp > 0:
            sp -= 1
            p  = stack[sp]
            k2 = _region_query(p, t, y, w, eps, lim, buf)
            if k2 >= min_samples:                           # p is a core point: expand
                for m in range(k2):
                    q = buf[m]
                    if labels[q] == UNVISITED:
                        stack[sp] = q
                        sp += 1
                        labels[q] = c
                    elif labels[q] == NOISE:
                        labels[q] = c
        c += 1
    return labels


def cluster_fdmt(fredda_cands, dt, disp, eps=1.0, min_samples=2):
    secs = np.ascontiguousarray(fredda_cands['secs'], dtype=np.float64)
    y    = np.ascontiguousarray(fredda_cands['dm'] * disp, dtype=np.float64)   # DM as delay [s]
    w    = np.maximum(fredda_cands['boxcar'] * dt, dt).astype(np.float64)

    order = np.argsort(secs, kind="stable")                 # sorted by time for the window search
    lab_s = _dbscan(secs[order], y[order], w[order], float(eps), int(min_samples))

    labels = np.empty_like(lab_s)
    labels[order] = lab_s                                   # back to the input order
    return labels


def cluster_mb(path, dt, eps=1.0, min_samples=2, udp=None, npackets=100, timeout=10.0):
    """Cluster each beam independently. Returns (cands, labels, beams).
    Cluster ids are unique across beams, -1 means noise.
    path is the candidate text file. If udp="HOST:PORT" is given, candidates are read
    from that socket instead and path is ignored (pass None)."""
    if udp is None:
        cands, disp = load_fredda(path, dt)
    else:
        cands, disp = load_udp(udp, dt, npackets, timeout)
    beams  = np.unique(cands['beamno'])
    labels = np.full(len(cands), -1, dtype=np.int64)
    offset = 0

    for b in beams:
        idx = np.flatnonzero(cands['beamno'] == b)
        if len(idx) < min_samples:
            continue
        lab  = cluster_fdmt(cands[idx], dt, disp, eps, min_samples)
        good = lab >= 0
        if good.any():
            labels[idx[good]] = lab[good] + offset
            offset += lab.max() + 1

    return cands, labels, beams


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Cluster FREDDA/FDMT candidates (per beam, numba).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-f",
                    "--file",
                    help="candidate file (S/N sampno secs boxcar idt dm beam mjd)")
    src.add_argument("-u",
                    "--udp",
                    metavar="HOST:PORT",
                    help="read candidates from UDP datagrams on HOST:PORT (multicast joined automatically)")
    p.add_argument("--npackets",
                    type=int,
                    default=0,
                    help="with --udp: stop after this many datagrams (default 0 = no limit)")
    p.add_argument("--timeout",
                    type=float,
                    default=10.0,
                    help="with --udp: stop after this many seconds without a datagram (default 10)")
    p.add_argument("--dt",
                    type=float, 
                    required=True,
                    help="sample time in milliseconds, e.g. 1.28")
    p.add_argument("--eps", 
                    type=float, 
                    default=1.0,
                    help="DBSCAN eps in boxcar widths (default 1)")
    p.add_argument("--min-samples", 
                    type=int, 
                    default=2,
                    help="DBSCAN min_samples (default 2)")
    p.add_argument("-o", 
                    "--outfile", 
                    type=str, 
                    default=2,
                    help="Clustered outputfile")
    args = p.parse_args()

    fredda_cands, labels, beams = cluster_mb(args.file,
                                             dt=args.dt * 1e-3,
                                             eps=args.eps,
                                             min_samples=args.min_samples,
                                             udp=args.udp,
                                             npackets=args.npackets,
                                             timeout=args.timeout)

    for b in beams:
        lab = labels[fredda_cands['beamno'] == b]
        print(f"beam {b}: {len(lab)} candidates -> {len(np.unique(lab[lab >= 0]))} clusters, "
              f"{(lab == -1).sum()} noise")
    print(f"total: {len(labels)} candidates -> {len(np.unique(labels[labels >= 0]))} clusters, "
          f"{(labels == -1).sum()} noise")

    valid = np.flatnonzero(labels >= 0)
    order = valid[np.lexsort((fredda_cands['sn'][valid], labels[valid]))]
    uniq, first, counts = np.unique(labels[order], return_index=True, return_counts=True)
    best = order[first + counts - 1]                        # highest S/N per cluster

    np.savetxt(f"{args.outfile}_clustered.txt",
               np.column_stack([fredda_cands['sn'][best],
                                fredda_cands['sampno'][best],
                                fredda_cands['secs'][best],
                                fredda_cands['boxcar'][best],
                                fredda_cands['idt'][best],
                                fredda_cands['dm'][best],
                                fredda_cands['beamno'][best],
                                fredda_cands['mjd'][best],
                                uniq,
                                counts]),
               fmt=["%.2f", "%d", "%.4f", "%d", "%d", "%.2f", "%d", "%.6f", "%d", "%d"],
               header="sn  sampno  secs  boxcar  idt  dm  beamno  mjd  cluster  ncands")

    noise = np.flatnonzero(labels < 0)
    np.savetxt(f"{args.outfile}_noise_candidates.txt",
               np.column_stack([noise,
                                fredda_cands['sn'][noise],
                                fredda_cands['secs'][noise],
                                fredda_cands['dm'][noise],
                                fredda_cands['boxcar'][noise],
                                fredda_cands['beamno'][noise]]),
               fmt=["%d", "%.2f", "%.4f", "%.2f", "%d", "%d"],
               header="index  sn  secs  dm  boxcar  beamno")
