import numpy as np

from neo import SpikeTrain, Event
import quantities as pq
from viziphant.rasterplot import rasterplot
from viziphant.events import add_event
##from jules
def plot_rasterplot(ax, times, events, events_toplot=[0], 
            window=[0, 6000], histogram_bins=0,
            remove_empty_trials=False,
            **kwargs):
    '''
    Uses viziphant rasterplot to plot raster along events

    Parameters:        
        - ax : list of matplotlib.axes.Axes 1 element if histogram_bins is 0, 2 else
        - times: list of spike times (in s)
        - events: list/array of events, in seconds or a quantities array. TRIALS WILL BE PLOTTED ACORDING TO EVENTS ORDER.
        TODO - events_toplot: list/array of events indices to display on the raster | Default: None (plots everything)
        - window: list/array of shape (2,): the raster will be plotted from events-window[0] to events-window[1] in ms | Default: [-1000,1000]
        - histogram_bins: number of bins in the histogram. if 0, only plots the raster | Default: 0
        - remove_empty_trials: boolean, if True does not use empty trials to compute psth

    Returns:
        - axes: matplotlib axes object
    '''

    at, atb = align_times(times, events, window=window, remove_empty_trials=remove_empty_trials)

    st_toplot = [SpikeTrain(at[ev]*pq.s, t_start=window[0]*pq.ms ,t_stop = window[1]*pq.ms) \
        for ev in at.keys()]

    rasterplot(st_toplot, s=3, c='black', axes=ax, histogram_bins=histogram_bins)
    add_event(ax, event=Event([0]*pq.s, labels=['']))
    for axx in ax:
        simple_xy_axes(axx)

    return ax


def align_times(times, events, b=2, window=[-1000,1000], remove_empty_trials=False):
    '''
    Parameters:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
    Returns:
        - aligned_t: dictionnaries where each key is an event in absolute time and value the times aligned to this event within window.
        - aligned_tb: a len(events) x window/b matrix where the spikes have been aligned, in counts.
    '''
    assert np.any(events), 'You provided an empty array of events!'
    t = np.sort(times)
    aligned_t = {}
    tbins=np.arange(window[0], window[1], b)
    aligned_tb = np.zeros((len(events), len(tbins))).astype(float)
    for i, e in enumerate(events):
        ts = t-e # ts: t shifted
        tsc = ts[(ts>=window[0]/1000)&(ts<=window[1]/1000)] # tsc: ts clipped
        if np.any(tsc) or not remove_empty_trials:
            aligned_t[e]=tsc.tolist()
            tscb = np.histogram(tsc*1000, bins=np.arange(window[0],window[1]+b,b))[0] # tscb: tsc binned
            aligned_tb[i,:] = tscb
        else:
            aligned_tb[i,:] = np.nan
    aligned_tb=aligned_tb[~np.isnan(aligned_tb).any(axis=1)]

    if not np.any(aligned_tb): aligned_tb = np.zeros((len(events), len(tbins)))

    return aligned_t, aligned_tb


def simple_xy_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def set_font_axes(ax, add_size=0, size_ticks=6, size_labels=8,
                  size_text=8, size_title=8, family='Arial'):
    '''add size to a bunch of stuff in your matplotlib axes to make it pretty'''
    
    if size_title is not None:
        ax.title.set_fontsize(size_title + add_size)

    if size_ticks is not None:
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=size_ticks + add_size)

    if size_labels is not None:

        ax.xaxis.label.set_fontsize(size_labels + add_size)
        ax.xaxis.label.set_fontname(family)

        ax.yaxis.label.set_fontsize(size_labels + add_size)
        ax.yaxis.label.set_fontname(family)

        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(size_labels + add_size)
            ax.zaxis.label.set_fontname(family)

    if size_text is not None:
        for at in ax.texts:
            at.set_fontsize(size_text + add_size)
            at.set_fontname(family)