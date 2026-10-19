# cryoPAF transient search pipeline


## The codebase is a proto-candidate pipeline for cryoPAF on Murriyang. 



## Contents
 
1. [Overview](#1-overview)
2. [Installation and dependencies](#2-Installation and dependencies)

## 1. Overview
 
The code base takes the raw candidate list written by FREDDA (FDMT-based single-pulse search) and reduces it to a short list of unique burst candidates, then checks each one against the known-pulsar catalogue.
 
`run_cand_pipe` is the main driver. It runs three stages in one command:
 
1. **Cluster** the raw candidates of each beam in (time, DM) space with a DBSCAN variant, and keep the highest-S/N candidate of each cluster. We do a normalisation in the DM-time space to avoid being baised towards the DM distance, since DBSCAN does a eucildean distance calculation. 
2. **Attach sky positions** to the surviving candidates using a beam-position table (beam number to RA/Dec). Currently, this is read through a text file. However, we'd need to have mechanism to read it through some API from `GARRIWANG`.
3. **Cross-match** against a known-pulsar catalogue by sky separation and DM. Candidates that match are labelled with the pulsar name.
 
```
raw FREDDA candidates ──► cluster_mb()  ──► best candidate per cluster
   (S/N sampno secs         (per beam,          │
    boxcar idt dm           numba DBSCAN)       ▼
    beamno mjd)                          attach_beam_coords()  ◄── beam position file
                                                │
                                                ▼
                                          crossmatch()  ◄── known-pulsar catalogue
                                                │
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                            *_matches.txt               *_unique.txt
```



## 2. Installation and dependencies

```bash
python -m venv cryo_venv --prompt cryopipe
```


```bash
pip install -e .
```   



