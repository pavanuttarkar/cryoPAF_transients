import re
import glob
import numpy as np
from matplotlib.pyplot import subplots, savefig, style

def create_subplots(rc_params='seaborn-v0_8-darkgrid'):
    style.use(rc_params)
    fig, ax = subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 4],
                     'wspace': 0,
                     'hspace': 0,},
    )
    return fig, ax


def read_rfi_stats(file='rfi_stats.txt'):
    return np.loadtxt(f"{file}")


def plot_rfi_stats(file='rfi_stats.txt',
                    bands=None,
                    outname='RFI_stats.pdf'):
    """
    Parameters
    ----------
    file : str
        Path to rfi stats file.
    bands : list of dict, optional
        Each dict draws a HORIZONTAL double-headed arrow spanning a
        given frequency range, at a given epoch, e.g.:
            {'freq_range': (2400, 2500), 'epoch': 65, 'text': 'WiFi'}
        'freq_range' : (fmin, fmax) in MHz -- the span the arrow covers
        'epoch'      : y-position (epoch number) where the arrow is drawn
        'text'       : label text, centered above the arrow
        Pass as many bands as you like; each is drawn independently.
        The full frequency axis is always shown -- no zooming.
    """
    fig, ax = create_subplots(rc_params='~/bin/matplotlib_rc_rfi')
    rfi     = np.loadtxt(f"{file}")
    f       = np.linspace(704, 4032, rfi.shape[1])
    n_epoch = len(rfi)

    ax[1].imshow(np.flipud(rfi),
                 aspect='auto',
                 cmap='plasma',
                 extent=[f[0], f[-1], 0, n_epoch],
                 vmin=0, vmax=0.9)
    ax[1].set_xlabel("Frequency (MHz)")
    ax[1].set_ylabel('Epochs (N)')
    ax[0].set_ylabel('Counts (N)')

    ax[0].plot(f, 
                np.nansum(rfi, axis=0), 
                color='white',
                )

    for a in ax:
        for spine_name, spine in a.spines.items():
            spine.set_visible(spine_name == 'left')
        a.tick_params(axis='both', which='major',
                       top=False, right=False, bottom=False, left=True)
        a.tick_params(axis='both', which='minor',
                       top=False, right=False, bottom=False, left=True)

    ax[1].tick_params(bottom=True)  # keep bottom ticks on the frequency axis

    # --- Horizontal double-headed arrows over given frequency ranges ---
    if bands:
        for b in bands:
            fmin, fmax = b['freq_range']
            epoch      = b['epoch']
            text       = b.get('text', '')

            ax[1].annotate(
                '',
                xy=(fmax, epoch),
                xytext=(fmin, epoch),
                arrowprops=dict(
                    arrowstyle='<->',
                    color='white',
                    lw=15.5,
                ),
            )
            ax[1].text(
                (fmin + fmax) / 2, epoch,
                text,
                color='white',
                fontsize=200,
                fontweight='bold',
                ha='center',
                va='bottom',
            )

    savefig(outname)



if __name__ == '__main__':
    plot_rfi_stats(
    file='rfi_mask_2024_2025.txt',
    bands=[
        {'freq_range':{1316,2050},  
        'epoch': 90, 
        'text': 'Clean band'},
        {'freq_range':{710,1200},  
        'epoch': 20, 
        'text': 'low band'},
        {'freq_range':{1100,1950},  
        'epoch': 30, 
        'text': 'high band'},
    ],
    )
