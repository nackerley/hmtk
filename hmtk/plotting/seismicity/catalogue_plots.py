#!/usr/bin/env/python
# -*- coding: UTF-8 -*-

"""
Collection of tools for plotting descriptive statistics of a catalogue
"""
import os
import itertools
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, Normalize
from matplotlib.offsetbox import AnchoredText

VALID_LINESTYLES = ['-', '--', '-.', ':']


def build_filename(filename, filetype='png', resolution=300):
    """
    Uses the input properties to create the string of the filename
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    filevals = os.path.splitext(filename)
    if filevals[1]:
        filetype = filevals[1][1:]
    if not filetype:
        filetype = 'png'

    filename = filevals[0] + '.' + filetype

    if not resolution:
        resolution = 300
    return filename, filetype, resolution


def _save_image(filename, filetype='png', resolution=300):
    """
    If filename is specified, saves the image
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    if filename is not None:
        filename, filetype, resolution = build_filename(filename,
                                                        filetype,
                                                        resolution)
        plt.savefig(filename, dpi=resolution, format=filetype)
    else:
        pass
    return


def _get_catalogue_bin_limits(catalogue, dmag):
    """
    Returns the magnitude bins corresponing to the catalogue
    """
    mag_bins = np.arange(
        float(np.floor(np.min(catalogue.data['magnitude']))) - dmag,
        float(np.ceil(np.max(catalogue.data['magnitude']))) + dmag,
        dmag)
    counter = np.histogram(catalogue.data['magnitude'], mag_bins)[0]
    idx = np.where(counter > 0)[0]
    mag_bins = mag_bins[idx[0]:idx[-1] + 3]
    return mag_bins


def plot_depth_histogram(catalogue, bin_width, normalisation=False,
                         bootstrap=None, colour='0.5', edgecolor='none',
                         filename=None, filetype='png', dpi=300):
    """
    Creates a histogram of the depths in the catalogue
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float bin_width:
        Width of the histogram for the depth bins
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample depth uncertainty choose number of samples
    """

    # Create depth range
    if len(catalogue.data['depth']) == 0:
        raise ValueError('No depths reported in catalogue!')
    depth_bins = np.arange(0.,
                           np.max(catalogue.data['depth']) + bin_width,
                           bin_width)
    depth_hist = catalogue.get_depth_distribution(depth_bins,
                                                  normalisation,
                                                  bootstrap)

    plt.bar(depth_bins[:-1], depth_hist,
            width=0.95*bin_width, color=colour, edgecolor='none')
    plt.xlabel('Depth (km)')
    if normalisation:
        plt.ylabel('Probability Mass Function')
    else:
        plt.ylabel('Count')

    _save_image(filename, filetype, dpi)


def plot_magnitude_depth_density(catalogue, mag_int, depth_int, logscale=False,
                                 normalisation=False, bootstrap=None,
                                 filename=None, filetype='png', dpi=300):
    """
    Creates a density plot of the magnitude and depth distribution
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float mag_int:
        Width of the histogram for the magnitude bins
    :param float depth_int:
        Width of the histogram for the depth bins
    :param bool logscale:
        Choose to scale the colours in a log-scale (True) or linear (False)
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample magnitude and depth uncertainties choose number of samples
    """
    if len(catalogue.data['depth']) == 0:
        raise ValueError('No depths reported in catalogue!')
    depth_bins = np.arange(0.,
                           np.max(catalogue.data['depth']) + depth_int,
                           depth_int)
    mag_bins = _get_catalogue_bin_limits(catalogue, mag_int)
    mag_depth_dist = catalogue.get_magnitude_depth_distribution(mag_bins,
                                                                depth_bins,
                                                                normalisation,
                                                                bootstrap)
    vmin_val = np.min(mag_depth_dist[mag_depth_dist > 0.])
    # Create plot
    if logscale:
        normaliser = LogNorm(vmin=vmin_val, vmax=np.max(mag_depth_dist))
    else:
        normaliser = Normalize(vmin=0, vmax=np.max(mag_depth_dist))

    plt.pcolor(mag_bins[:-1],
               depth_bins[:-1],
               mag_depth_dist.T,
               norm=normaliser)
    plt.xlabel('Magnitude')
    plt.ylabel('Depth (km)')
    plt.xlim(mag_bins[0], mag_bins[-1])
    plt.ylim(depth_bins[0], depth_bins[-1])
    plt.colorbar()
    if normalisation:
        plt.title('Magnitude-Depth Density')
    else:
        plt.title('Magnitude-Depth Count')

    _save_image(filename, filetype, dpi)


def plot_magnitude_time_scatter(catalogue, plot_error=False, filename=None,
                                filetype='png', dpi=300, fmt_string='o'):
    """
    Creates a simple scatter plot of magnitude with time
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param bool plot_error:
        Choose to plot error bars (True) or not (False)
    :param str fmt_string:
        Symbology of plot
    """

    dtime = catalogue.get_decimal_time()
    if len(catalogue.data['sigmaMagnitude']) == 0:
        print('Magnitude Error is missing - neglecting error bars!')
        plot_error = False

    if plot_error:
        plt.errorbar(dtime,
                     catalogue.data['magnitude'],
                     xerr=None,
                     yerr=catalogue.data['sigmaMagnitude'],
                     fmt=fmt_string)
    else:
        plt.plot(dtime, catalogue.data['magnitude'], fmt_string)
    plt.xlabel('Year')
    plt.ylabel('Magnitude')
    plt.title('Magnitude-Time Plot')

    _save_image(filename, filetype, dpi)


def plot_magnitude_time_density(catalogue, mag_int, time_int,
                                normalisation=False, bootstrap=None,
                                filename=None, filetype='png', dpi=300,
                                completeness=None):
    """
    Creates a plot of magnitude-time density
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float mag_int:
        Width of the histogram for the magnitude bins
    :param float time_int:
        Width of the histogram for the time bin (in decimal years)
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample magnitude and depth uncertainties choose number of samples
    """

    # Create the magnitude bins
    if isinstance(mag_int, np.ndarray) or isinstance(mag_int, list):
        mag_bins = mag_int
    else:
        mag_bins = np.arange(
            np.min(catalogue.data['magnitude']),
            np.max(catalogue.data['magnitude']) + 0.5*mag_int,
            mag_int)
    # Creates the time bins
    if isinstance(time_int, np.ndarray) or isinstance(time_int, list):
        time_bins = time_int
    else:
        time_bins = np.arange(
            float(np.min(catalogue.data['year'])),
            float(np.max(catalogue.data['year'])) + 1.,
            float(time_int))
    # Get magnitude-time distribution
    mag_time_dist = catalogue.get_magnitude_time_distribution(
        mag_bins,
        time_bins,
        normalisation,
        bootstrap)
    # Get smallest non-zero value
    vmin_val = np.min(mag_time_dist[mag_time_dist > 0.])
    # Create plot
    plt.pcolor(time_bins,
               mag_bins,
               mag_time_dist.T,
               norm=LogNorm(vmin=vmin_val, vmax=np.max(mag_time_dist)))
    plt.ylabel('Magnitude')
    plt.xlim(time_bins[0], time_bins[-1])
    plt.yticks(plt.yticks()[0][:-1])

    if normalisation:
        plt.colorbar(label='Event Density', shrink=0.9)
    else:
        plt.colorbar(label='Event Count', shrink=0.9)
    # Overlay completeness
    if completeness is not None:
        _plot_completeness(completeness)

    _save_image(filename, filetype, dpi)


def plot_mag_time_density_slices(
        catalogue, completeness_tables, slice_key, slice_ids,
        mag_bin=0.1, time_bin=1):
    """
    Magnitude-time density plots on sub-catalogues, where `slice_key` and
    `slice_ids` determine how the sub-catalouges are formed.
    """

    fig, axes = plt.subplots(len(slice_ids), 1,
                             figsize=(8, 3*len(slice_ids)), sharex=True)
    fig.subplots_adjust(hspace=0)
    for ax, slice_id, completeness_tables_slice \
            in zip(axes, slice_ids, completeness_tables):
        plt.sca(ax)
        ax_label = '%s %d' % (slice_key, slice_id)
        ax.add_artist(AnchoredText(ax_label, loc=3, frameon=False))

        catalogue_slice = deepcopy(catalogue)
        in_slice = catalogue_slice.data[slice_key] == slice_id
        catalogue_slice.select_catalogue_events(in_slice)
        plot_magnitude_time_density(
            catalogue_slice, mag_bin, time_bin,
            completeness=completeness_tables_slice)


def _plot_completeness(completeness_tables):
    """
    Overlay one or more completeness tables on a magnitude-time plot.
    """
    completeness_tables = np.array(completeness_tables)
    if len(completeness_tables.shape) < 3:
        completeness_tables.reshape((1, completeness_tables.shape))

    linestyle_cycler = itertools.cycle(VALID_LINESTYLES)
    for data in completeness_tables:
        data = np.flipud(data[np.argsort(data[:, 0]), :])
        start = [data[-1, 0], plt.gca().get_ylim()[1]]
        end = [plt.gca().get_xlim()[1], data[0, 1]]
        data = np.vstack((end, data, start))
        plt.step(data[:, 0], data[:, 1], where='pre',
                 linewidth=2, linestyle=next(linestyle_cycler), color='brown')


def get_completeness_adjusted_table(catalogue, completeness, dmag, end_year):
    """
    Counts the number of earthquakes in each magnitude bin and normalises
    the rate to annual rates, taking into account the completeness
    """
    inc = 1E-7
    # Find the natural bin limits
    mag_bins = _get_catalogue_bin_limits(catalogue, dmag)
    obs_time = end_year - completeness[:, 0] + 1.
    obs_rates = np.zeros_like(mag_bins)
    n_comp = np.shape(completeness)[0]
    for iloc in range(0, n_comp, 1):
        low_mag = completeness[iloc, 1]
        comp_year = completeness[iloc, 0]
        if iloc == n_comp - 1:
            idx = np.logical_and(
                catalogue.data['magnitude'] >= low_mag - (dmag / 2.),
                catalogue.data['year'] >= comp_year)
            high_mag = mag_bins[-1] + dmag
            obs_idx = mag_bins >= (low_mag - dmag / 2.)
        else:
            high_mag = completeness[iloc + 1, 1]
            mag_idx = np.logical_and(
                catalogue.data['magnitude'] >= low_mag - dmag / 2.,
                catalogue.data['magnitude'] < high_mag)

            idx = np.logical_and(mag_idx,
                                 catalogue.data['year'] >= comp_year - inc)
            obs_idx = np.logical_and(mag_bins >= low_mag - dmag / 2.,
                                     mag_bins < high_mag + dmag)
        temp_rates = np.histogram(catalogue.data['magnitude'][idx],
                                  mag_bins[obs_idx])[0]
        # print mag_bins[obs_idx], temp_rates
        temp_rates = temp_rates.astype(float) / obs_time[iloc]
        if iloc == n_comp - 1:
            # TODO This hack seems to fix the error in Numpy v.1.8.1
            obs_rates[np.where(obs_idx)[0]] = temp_rates
        else:
            obs_rates[obs_idx[:-1]] = temp_rates
    selector = np.where(obs_rates > 0.)[0]
    mag_bins = mag_bins[selector[0]:selector[-1] + 1]
    obs_rates = obs_rates[selector[0]:selector[-1] + 1]
    # Get cumulative rates
    cum_rates = np.array([sum(obs_rates[iloc:])
                          for iloc in range(0, len(obs_rates))])
    out_idx = cum_rates > 0.
    # print mag_bins[out_idx], obs_rates[out_idx], cum_rates[out_idx]
    return np.column_stack([mag_bins[out_idx],
                            obs_rates[out_idx],
                            cum_rates[out_idx],
                            np.log10(cum_rates[out_idx])])


def plot_observed_recurrence(catalogue, completeness, dmag, end_year=None,
                             filename=None, filetype='png', dpi=300):
    """
    Plots the observed recurrence taking into account the completeness
    """
    # Get completeness adjusted recurrence table
    if isinstance(completeness, float):
        # Unique completeness
        completeness = np.array([[np.min(catalogue.data['year']),
                                  completeness]])
    if not end_year:
        end_year = np.max(catalogue.data['year'])
    recurrence = get_completeness_adjusted_table(catalogue,
                                                 completeness,
                                                 dmag,
                                                 end_year)

    plt.semilogy(recurrence[:, 0], recurrence[:, 1], 'bo')
    plt.semilogy(recurrence[:, 0], recurrence[:, 2], 'rs')
    plt.xlim([recurrence[0, 0] - 0.1, recurrence[-1, 0] + 0.1])
    plt.xlabel('Magnitude')
    plt.ylabel('Annual Rate')
    plt.legend(['Incremental', 'Cumulative'])

    _save_image(filename, filetype, dpi)


def plot_depth_distance(catalogue, limits, ordinate, name=None,
                        colour='black', size=4):
    """
    Produces a "side-view" of a portion of a catalogue. Subcatalogue selection
    is currently a simple rectangle of latitudes and longitudes. Ordinates
    supported are currently 'latitude' or 'longitude'.

    :param catalogue: instance of :class:`hmtk.seismicity.catalogue.Catalogue`
    :param tuple limits: lat_min, lat_max, lon_min, lon_max
    :param string ordinate: distance to plot on x-axis
    """

    assert ordinate in ['latitude', 'longitude']

    subcatalogue = deepcopy(catalogue)

    lat_min, lat_max, lon_min, lon_max = limits
    in_box = ((subcatalogue.data['latitude'] >= lat_min) &
              (subcatalogue.data['latitude'] <= lat_max) &
              (subcatalogue.data['longitude'] >= lon_min) &
              (subcatalogue.data['longitude'] <= lon_max))
    subcatalogue.purge_catalogue(in_box)

    if colour == 'magnitude':
        colour = subcatalogue.data['magnitude']
        cmap = 'jet'
    else:
        cmap = 'none'
    plt.scatter(subcatalogue.data[ordinate], subcatalogue.data['depth'],
                c=colour, s=size, cmap=cmap, edgecolor='none')

    ax = plt.gcf().gca()
    if ordinate == 'latitude':
        ax_label = u'Longitude: %g°-%g°' % (lon_min, lon_max)
        plt.xlabel(u'Latitude (°)')
        plt.xlim(lat_min, lat_max)
    else:
        ax_label = u'Latitude: %g°-%g°' % (lat_min, lat_max)
        plt.xlabel(u'Longitude (°)')
        plt.xlim(lon_min, lon_max)
    if name is not None:
        ax_label = name + '\n' + ax_label
    ax.add_artist(AnchoredText(ax_label, loc=2, frameon=False))
