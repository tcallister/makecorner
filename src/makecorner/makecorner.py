import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import style
import os
style.use(os.path.dirname(os.path.realpath(__file__))+'/../../plotting.mplstyle')

def getBounds(data):

    """
    Helper function to obtain 90% credible bounds from a list of samples
    Invoked by plot_corner to create labels on 1D posteriors

    Parameters
    ----------
    data : list or numpy.array
        1D array of samples

    Returns
    -------
    med : float
        Median of samples
    upperError : float
        Difference between 95th and 50th percentiles of data
    lowerError : float
        Difference between 50th and 5th percentiles of data
    """

    # Transform to a numpy arry
    data = np.array(data)

    # Get median, 5% and 95% quantiles
    med = np.median(data)
    upperLim = np.sort(data)[int(0.95*data.size)]
    lowerLim = np.sort(data)[int(0.05*data.size)]
 
    # Turn quantiles into upper and lower uncertainties
    upperError = upperLim - med
    lowerError = med - lowerLim
    
    return med,upperError,lowerError
    
def corner(
        plot_data,
        *,
        color='#1f78b4',
        hist_alpha=0.7,
        bins=20,
        labelsize=14,
        ticklabelsize=10,
        titlesize=14,
        show_bounds=True,
        logscale=False,
        vmax=None,
        figsize=None,
        hspace=0.1,
        wspace=0.1,
        contour_levels=None,
        contour_kde_args={},
        contour_plot_args={'colors':'black'}):

    """
    Helper function to generate nice-looking corner plots. The primary input, `plot_data`,
    should be a dictionary containing data to be plotted. Every item in this dictionay corresponds 
    to a data column, and should itself be a dictionary possessing the following keys:

    * `data`: Posterior sample values
    * `plot_bounds`: Tuple of min/max values to display on plot
    * `label`: A latex string for axis labeling

    e.g.

    ```
    plot_data = {
        'x': {'data': [...], 'plot_bounds': (-1, 1), 'label': r"$x$"},
        'y': {'data': [...], 'plot_bounds': (0, 1), 'label': r"$y$"}
        }
    ```

    Parameters
    ----------
    plot_data : dict
        Dictionary containing data to plot; see above
    color : str (optional)
        Hexcode defining plot color. Default `'#1f78b4'` (blue).
    hist_alpha : float (optional)
        Defines transparency of 1D histograms. Default `0.7`.
    bins : int (optional)
        Defines number of 1D histogram bins and 2D hexbins to use. Default `20`.
    labelsize : int (optional)
        Defines fontsize of axis labels. Default `14` 
    ticklabelsize : int (optional)
        Defines fontsize of axis ticklabels. Default `10`.
    titlesize : int (optional)
        Defines fontsize of plot titles quoting marginal credible intervals. Default `14`.
    show_bounds : bool (optional)
        If True, will quote marginal 95% credible intervals above 1D histograms. Default `True`.
    logscale : bool (optional)
        If True, a logarithmic color scale is adopted for 2D posteriors. Default `False`.
    vmax : None or float (optional)
        User-specified maximum for 2D colorscale. Default `None`.
    figsize : None or tuple (optional)
        User-specified size of figure to create. If `None`, figure defaults to size `(2*ndim, 2*ndim)`,
        where `ndim` is the number of data columns being plotted. Default `None`.
    hspace : float (optional)
        Float that adjusts the vertical spacing of subplots. Default `0.1`.
    wspace : float (optional)
        Float that adjusts the horizontal spacing of subplots. Default `0.1`.
    contour_levels : None, list, or tuple (optional)
        If not `None`, then defines probabilities at which contour levels will be drawn in 2D subplots.
        Default `None`.
    contour_kde_args : None or dict (optional)
        Keyword arguments provided to `scipy.stats.gaussian_kde` as a step in creating contours.
        Can be used, e.g. to adjust KDE bandwidth. Default `None`.
    contour_plot_args : None or dict (optional)
        Keyword arguments provided to `matplotlib.pyplot.contour`. Can be used to adjust contour linestyles,
        etc. Default `{'colors':'black'}`.
        

    Returns
    -------
    fig : matplotlib.figure.Figure
        Populated figure
    """

    keys = list(plot_data)    
    ndim = len(keys)

    if figsize is None:
        figsize = (2*ndim, 2*ndim)
    fig = plt.figure(figsize=figsize)
    
    if logscale==True:
        hexscale='log'
    else:
        hexscale=None

    # Define a linear color map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", color])
    
    # Loop across dimensions that we want to plot
    for i, key in enumerate(keys):
       
        # Plot the marginal 1D posterior (i.e. top of a corner plot column)
        ax = fig.add_subplot(ndim, ndim, int(1+(ndim+1)*i))
        
        ax.hist(plot_data[key]['data'],
                bins=np.linspace(plot_data[key]['plot_bounds'][0], plot_data[key]['plot_bounds'][1], bins),
                rasterized=True,
                color=color,
                alpha=hist_alpha,
                density=True,
                zorder=0)
        ax.hist(plot_data[key]['data'],
                bins=np.linspace(plot_data[key]['plot_bounds'][0], plot_data[key]['plot_bounds'][1], bins),
                histtype='step',
                color='black',
                density=True,
                zorder=2)
        ax.grid(True, dashes=(1, 3))
        ax.set_xlim(plot_data[key]['plot_bounds'][0], plot_data[key]['plot_bounds'][1])
        if show_bounds:
            ax.set_title(r"${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(*getBounds(plot_data[key]['data'])), fontsize=titlesize)

        # Turn off tick labels if this isn't the first dimension
        if i!=0:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='y', which='major', labelsize=ticklabelsize)

        # If this is the last dimension add an x-axis label
        if i == ndim-1:
            ax.set_xlabel(plot_data[key]['label'], fontsize=labelsize)
            ax.tick_params(axis='x', which='major', labelsize=ticklabelsize)
            
        # If not the last dimension, loop across other variables and fill in the rest of the column with 2D plots
        else:
            
            ax.set_xticklabels([])
            for j, k in enumerate(keys[i+1:]):
                
                # Make a 2D density plot
                ax = fig.add_subplot(ndim, ndim, int(1+(ndim+1)*i + (j+1)*ndim))
                
                ax.hexbin(
                    plot_data[key]['data'],
                    plot_data[k]['data'],
                    cmap=cmap,
                    mincnt=1,
                    gridsize=bins,
                    bins=hexscale,
                    rasterized=True,
                    extent=(
                        plot_data[key]['plot_bounds'][0],
                        plot_data[key]['plot_bounds'][1],
                        plot_data[k]['plot_bounds'][0],
                        plot_data[k]['plot_bounds'][1]),
                    linewidths=(0,),
                    zorder=0,
                    vmax=vmax)

                # Plot contours if requested
                if contour_levels is not None:

                    # Set up KDE and evaluate over grid
                    kde = gaussian_kde([plot_data[key]['data'], plot_data[k]['data']], **contour_kde_args)
                    xgrid = np.linspace(plot_data[key]['plot_bounds'][0], plot_data[key]['plot_bounds'][1], 100)
                    ygrid = np.linspace(plot_data[k]['plot_bounds'][0], plot_data[k]['plot_bounds'][1], 100)
                    X, Y = np.meshgrid(xgrid, ygrid)
                    pdf = kde([X.reshape(-1), Y.reshape(-1)])

                    # Interpolate levels onto PDF grid
                    sorted_pdf = np.sort(pdf)[::-1]
                    cdf = np.cumsum(sorted_pdf)
                    cdf /= cdf[-1]

                    # Contour plot
                    levels = [np.interp(l, cdf, sorted_pdf) for l in contour_levels]
                    ax.contour(xgrid, ygrid, pdf.reshape(X.shape), levels=np.sort(levels), **contour_plot_args)

                
                # Set plot bounds
                ax.set_xlim(plot_data[key]['plot_bounds'][0], plot_data[key]['plot_bounds'][1])
                ax.set_ylim(plot_data[k]['plot_bounds'][0], plot_data[k]['plot_bounds'][1])
                ax.grid(True, dashes=(1, 3))
                
                # If still in the first column, add a y-axis label
                if i==0:
                    ax.set_ylabel(plot_data[k]['label'], fontsize=labelsize)
                    ax.tick_params(axis='y', which='major', labelsize=ticklabelsize)
                else:
                    ax.set_yticklabels([])
               
                # If on the last row, add an x-axis label
                if j==ndim-i-2:
                    ax.set_xlabel(plot_data[key]['label'], fontsize=labelsize)
                    ax.tick_params(axis='x', which='major', labelsize=ticklabelsize)
                else:
                    ax.set_xticklabels([])
                    
    plt.tight_layout()    
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    return fig
