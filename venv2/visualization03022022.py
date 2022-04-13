import numpy as np
import matplotlib.pyplot as plt
import math


def rasters(data,sorted_array,epoch_offset, subplots=(5, 6), fig=None, axes=None, figsize=(9*1.5, 5*1.5),
            max_spikes=7000, style='black', **scatter_kw):
    """
    Plots a series of spike raster plots.

    Parameters
    ----------
    data : SpikeData instance
        Multi-trial spike data.
    subplots : tuple
        2-element tuple specifying number of rows and columns of subplots.
        epoch_offset=what number of ms did you epoch the before the trial or stim presentation of interest happened
    fig : Maplotlib Figure or None
        Figure used for plotting.
    axes : ndarray of Axes objects or None
        Axes used for plotting.
    figsize : tuple
        Dimensions of figure size.
    max_spikes : int
        Maximum number of spikes to plot on a raster. Spikes are randomly
        subsampled above this limit.
    style : string
        Either ('black' or 'white') specifying background color.
    **scatter_kw
        Additional keyword args are passed to matplotlib.pyplot.scatter

    Returns
    -------
    fig : matplotlib.Figure instance
    axes : ndarray of matplotlib.Axes objects
    """

    trials, times, neurons = data.trials, data.spiketimes, data.neurons

    background = 'k' if style == 'black' else 'w'
    foreground = 'w' if style == 'black' else 'k'

    scatter_kw.setdefault('s', 1)
    scatter_kw.setdefault('lw', 0)

    # handle coloring of rasters
    if 'c' not in scatter_kw:
        scatter_kw.setdefault('color', foreground)
        c = None
    else:
        c = scatter_kw.pop('c')

    if axes is None:
        fig, axes = plt.subplots(*subplots, figsize=figsize)

    for n, ax in enumerate(axes.ravel()):

        # select spikes for neuron n
        idx = np.where(neurons == (n+1))[0]

        # turn off axis if there are no spikes
        if len(idx) == 0:
            ax.axis('off')
            continue

        # subsample spikes
        elif len(idx) > max_spikes:
            idx = np.random.choice(idx, size=max_spikes, replace=False)

        # make raster plot
        if c is not None:
            ax.scatter(times[idx], trials[idx], c=c[idx], **scatter_kw)
            #ax.yaxis.set_major_locator(MaxNLocator(5))

            #ax.set_xticks([0, 200, 400, 600, 800])
            #ax.set_xticklabels([-400, -200, 0, 200, 400])
            #ax.set_xticks(np.arange(min(times), max(times) + 1, 200))


        else:
            ax.scatter(times[idx], trials[idx], **scatter_kw)
            #ax.set_xlabel('milliseconds')

            #ax.set_xticks([0, 200, 400, 600, 800])
            #ax.set_xticklabels([-400, -200, 0, 200, 400])



        # format axes
        #ax.plot(sorted_array[:, 0], sorted_array[:, 1], c="red", marker='.', linestyle=':')
        ax.set_title('site {}'.format((n+1)), color=foreground)
        ax.set_facecolor(background)
        ax.set_xticks(np.arange(math.floor(min(times)), math.ceil(max(times)+math.floor(epoch_offset)), math.ceil(max(times))/4))
        ax.set_xticklabels(np.arange(math.floor(min(times))-math.floor(epoch_offset), math.ceil(max(times)+math.floor(epoch_offset))-math.floor(epoch_offset), math.ceil(max(times))/4), fontsize=6)

        #ax.set_xticks(np.arange(math.floor(min(times)), math.floor(max(times)), 200))
        ax.set_yticks(np.arange(math.floor(min(sorted_array[:, 1])), math.ceil(max(sorted_array[:, 1])),
                               math.ceil(max(sorted_array[:, 1] / 5))))

        #ax.set_yticklabels(np.arange(math.floor(min(sorted_array[:, 0])), math.floor(max(sorted_array[:, 0])), 1000))
        #ax.set_yticks(np.arange(math.floor(min(sorted_array[:,1])), math.floor(max(sorted_array[:,1])), math.ceil(max(sorted_array[:,1])/5)))


        if min(sorted_array[:,0])<0:
            ax.set_yticklabels(np.arange(math.floor(min(sorted_array[:,0])), math.floor(max(sorted_array[:,0])), math.ceil((max(sorted_array[:,0])-min(sorted_array[:,0]))/5)))
        elif max(sorted_array[:, 0]) == 0.0:
            ax.set_yticklabels(np.arange(math.floor(min(sorted_array[:,1])), math.ceil(max(sorted_array[:,1])), math.ceil(max(sorted_array[:,1])/5)))

        else:
            ax.set_yticklabels(np.arange(math.floor(min(sorted_array[:,0])), math.ceil(max(sorted_array[:,0])), math.ceil(max(sorted_array[:,0])/5)))
        ax.set_xlabel('milliseconds')
        ax.set_ylabel('LR Time (ms)')


        #ax.set_xticklabels([i + 100 for i in times])
        #ax.set_xticks([np.arange(min(times), max(times)+1, 100)])
        #ax.set_xticklabels([np.arange(min(times), max(times)+1, 100)])
        #ax.set_yticks([trials])
        ax.set_xlim([data.tmin, data.tmax])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if fig is not None:
        fig.tight_layout()
        fig.patch.set_facecolor(background)

    return fig, axes


def binned_heatmap(binned, subplots=(5, 6), figsize=(9*1.5, 5*1.5), **kwargs):

    kwargs.setdefault('aspect', 'auto')

    fig, axes = plt.subplots(*subplots, figsize=figsize)

    for ax, n in zip(axes.ravel(), range(binned.shape[-1])):
        ax.imshow(binned[:, :, n], **kwargs)
        ax.set_facecolor('k')
        ax.set_title('neuron {}'.format(n), color='w')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes.ravel()[n:]:
        ax.axis('off')

    fig.tight_layout()
    fig.patch.set_facecolor('k')

    return fig, axes
def psth_plots(data,sorted_array,NBINS, TMIN, TMAX, combinedTrials, plot_color, epoch_offset, subplots=(5, 6), fig=None, axes=None, figsize=(9*1.5, 5*1.5),
            max_spikes=7000, style='black', **scatter_kw):
    """
    Plots a series of spike PSTH plots to complement the original raster plot functionality.

    Parameters
    ----------
    data : SpikeData instance
        Multi-trial spike data.
    subplots : tuple
        2-element tuple specifying number of rows and columns of subplots.
    fig : Maplotlib Figure or None
        Figure used for plotting.
    axes : ndarray of Axes objects or None
        Axes used for plotting.
    figsize : tuple
        Dimensions of figure size.
    max_spikes : int
        Maximum number of spikes to plot on a raster. Spikes are randomly
        subsampled above this limit.
    style : string
        Either ('black' or 'white') specifying background color.
    **scatter_kw
        Additional keyword args are passed to matplotlib.pyplot.scatter

    Returns
    -------
    fig : matplotlib.Figure instance
    axes : ndarray of matplotlib.Axes objects
    """

    trials, times, neurons = data.trials, data.spiketimes, data.neurons

    background = 'k' if style == 'black' else 'w'
    foreground = 'w' if style == 'black' else 'k'

    scatter_kw.setdefault('s', 1)
    scatter_kw.setdefault('lw', 0)

    # handle coloring of rasters
    if 'c' not in scatter_kw:
        scatter_kw.setdefault('color', foreground)
        c = None
    else:
        c = scatter_kw.pop('c')

    if axes is None:
        fig, axes = plt.subplots(*subplots, figsize=figsize)

    for n, ax in enumerate(axes.ravel()):

        # select spikes for neuron n
        idx = np.where(neurons == (n+1))[0]

        # turn off axis if there are no spikes
        if len(idx) == 0:
            ax.axis('off')
            continue

        # subsample spikes
        elif len(idx) > max_spikes:
            idx = np.random.choice(idx, size=max_spikes, replace=False)

        # make raster plot
        if c is not None:
            hist, edges = np.histogram(
                times[idx],
                bins=NBINS,
                range=(0, 10 * NBINS),
                density=False)

            tvec = np.linspace(TMIN, TMAX, NBINS)

            #ax.plot(tvec, ((hist / max(combinedTrials) + 1)), plot_color)
            ax.plot(tvec, ((hist)), plot_color)

            #ax.yaxis.set_major_locator(MaxNLocator(5))

            #ax.set_xticks([0, 200, 400, 600, 800])
            #ax.set_xticklabels([-400, -200, 0, 200, 400])
            #ax.set_xticks(np.arange(min(times), max(times) + 1, 200))


        else:
            hist, edges = np.histogram(
                times[idx],
                bins=NBINS,
                range=(0, 10 * NBINS),
                density=False)
            tvec = np.linspace(TMIN, TMAX, NBINS)
            #ax.plot(tvec, ((hist / max(combinedTrials) + 1)), plot_color)
            ax.plot(tvec, ((hist)), plot_color)



        # format axes
        #ax.plot(sorted_array[:, 0], sorted_array[:, 1], c="red", marker='.', linestyle=':')
        ax.set_title('site {}'.format((n+1)), color=foreground)
        ax.set_facecolor(background)

        ax.set_xlabel('milliseconds')
        ax.set_xticks(np.arange(math.floor(0), math.ceil(TMAX), math.ceil(TMAX / 3)))
        ax.set_xticklabels(np.arange(math.floor(0) - math.floor(epoch_offset), math.ceil(TMAX) - math.floor(epoch_offset), math.ceil(TMAX / 3)) )

        ax.set_ylabel('Number of Spikes')


        #ax.set_xticklabels([i + 100 for i in times])
        #ax.set_xticks([np.arange(min(times), max(times)+1, 100)])
        #ax.set_xticklabels([np.arange(min(times), max(times)+1, 100)])
        #ax.set_yticks([trials])
        ax.set_xlim([data.tmin, data.tmax])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if fig is not None:
        fig.tight_layout()
        fig.patch.set_facecolor(background)

    return fig, axes
def rasterSite(data,sorted_array, siteschosen, fig=None, max_spikes=7000, style='black', **scatter_kw):
    """
    Plots a series of spike raster plots by SITE chosen.

    Parameters
    ----------
    data : SpikeData instance
        Multi-trial spike data.
    subplots : tuple
        2-element tuple specifying number of rows and columns of subplots.
    fig : Maplotlib Figure or None
        Figure used for plotting.
    axes : ndarray of Axes objects or None
        Axes used for plotting.
    figsize : tuple
        Dimensions of figure size.
    max_spikes : int
        Maximum number of spikes to plot on a raster. Spikes are randomly
        subsampled above this limit.
    style : string
        Either ('black' or 'white') specifying background color.
    **scatter_kw
        Additional keyword args are passed to matplotlib.pyplot.scatter

    Returns
    -------
    fig : matplotlib.Figure instance
    axes : ndarray of matplotlib.Axes objects
    """

    trials, times, neurons = data.trials, data.spiketimes, data.neurons

    background = 'k' if style == 'black' else 'w'
    foreground = 'w' if style == 'black' else 'k'

    scatter_kw.setdefault('s', 1)
    scatter_kw.setdefault('lw', 0)

    # handle coloring of rasters
    if 'c' not in scatter_kw:
        scatter_kw.setdefault('color', foreground)
        c = None
    else:
        c = scatter_kw.pop('c')

    # if axes is None:
    #     fig, axes = plt.subplots(*subplots, figsize=figsize)

    for n in siteschosen:

        # select spikes for neuron n
        idx = np.where(neurons == n)[0]

        # turn off axis if there are no spikes
        if len(idx) == 0:
            ax.axis('off')
            continue

        # subsample spikes
        elif len(idx) > max_spikes:
            idx = np.random.choice(idx, size=max_spikes, replace=False)

        # make raster plot
        if c is not None:
            plt.scatter(times[idx], trials[idx], c=c[idx], **scatter_kw)
            #ax.yaxis.set_major_locator(MaxNLocator(5))

            #ax.set_xticks([0, 200, 400, 600, 800])
            #ax.set_xticklabels([-400, -200, 0, 200, 400])
            #ax.set_xticks(np.arange(min(times), max(times) + 1, 200))


        else:
            plt.scatter(times[idx], trials[idx], **scatter_kw)
            #ax.set_xlabel('milliseconds')

            #ax.set_xticks([0, 200, 400, 600, 800])
            #ax.set_xticklabels([-400, -200, 0, 200, 400])



        # format axes
        #ax.plot(sorted_array[:, 0], sorted_array[:, 1], c="red", marker='.', linestyle=':')
        plt.title('neuron {}'.format(n), color=foreground)
        #ax.set_facecolor('White')
        plt.xticks(np.arange(math.floor(min(times)), math.ceil(max(times)), math.ceil(max(times))/2.5))
        #ax.set_xticklabels(np.arange(math.floor(min(times))-200, math.ceil(max(times))-200, math.ceil(max(times))/4), fontsize=6)

        #ax.set_xticks(np.arange(math.floor(min(times)), math.floor(max(times)), 200))
        plt.yticks(np.arange(math.floor(min(sorted_array[:, 1])), math.ceil(max(sorted_array[:, 1])),
                               math.ceil(max(sorted_array[:, 1] / 5))))

        #ax.set_yticklabels(np.arange(math.floor(min(sorted_array[:, 0])), math.floor(max(sorted_array[:, 0])), 1000))
        #ax.set_yticks(np.arange(math.floor(min(sorted_array[:,1])), math.floor(max(sorted_array[:,1])), math.ceil(max(sorted_array[:,1])/5)))


        if min(sorted_array[:,0])<0:
            plt.yticks(np.arange(math.floor(min(sorted_array[:,0])), math.floor(max(sorted_array[:,0])), math.ceil((max(sorted_array[:,0])-min(sorted_array[:,0]))/5)))
        elif max(sorted_array[:, 0]) == 0.0:
            plt.yticks(np.arange(math.floor(min(sorted_array[:,1])), math.ceil(max(sorted_array[:,1])), math.ceil(max(sorted_array[:,1])/5)))

        else:
            plt.yticks(np.arange(math.floor(min(sorted_array[:,0])), math.ceil(max(sorted_array[:,0])), math.ceil(max(sorted_array[:,0])/5)))
        plt.xlabel('milliseconds')
        plt.ylabel('Lick Release Time (ms)')


        #ax.set_xticklabels([i + 100 for i in times])
        #ax.set_xticks([np.arange(min(times), max(times)+1, 100)])
        #ax.set_xticklabels([np.arange(min(times), max(times)+1, 100)])
        #ax.set_yticks([trials])
        plt.xlim([data.tmin, data.tmax])
        # for spine in ax.spines.values():
        #     spine.set_visible(False)

    # if fig is not None:
    #     fig.tight_layout()
    #     #patches.set_facecolor(background)

    return fig

