#!/usr/bin/env/python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (c) 2010-2013, GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>
#
# DISCLAIMER
#
# The software Hazard Modeller's Toolkit (hmtk) provided herein
# is released as a prototype implementation on behalf of
# scientists and engineers working within the GEM Foundation (Global
# Earthquake Model).
#
# It is distributed for the purpose of open collaboration and in the
# hope that it will be useful to the scientific, engineering, disaster
# risk and software design communities.
#
# The software is NOT distributed as part of GEM’s OpenQuake suite
# (http://www.globalquakemodel.org/openquake) and must be considered as a
# separate entity. The software provided herein is designed and implemented
# by scientific staff. It is not developed to the design standards, nor
# subject to same level of critical review by professional software
# developers, as GEM’s OpenQuake software suite.
#
# Feedback and contribution to the software is welcome, and can be
# directed to the hazard scientific staff of the GEM Model Facility
# (hazard@globalquakemodel.org).
#
# The Hazard Modeller's Toolkit (hmtk) is therefore distributed WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# The GEM Foundation, and the authors of the software, assume no
# liability for use of the software.

'''
Module :mod: 'hmtk.plotting.seismicity.completeness.plot_stepp_1971'
creates plot to illustrate outcome of Stepp (1972) method for completeness
analysis
'''
import os.path
import itertools
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from hmtk.seismicity.completeness.comp_stepp_1971 import Stepp1971

# markers which can be filled or empty
VALID_MARKERS = ['s', 'o', '^', 'D', 'p', 'h', '8',
                 '*', 'd', 'v', '<', '>', 'H']


def create_stepp_plot(model, filename=None, filetype='png', filedpi=300,
                      ax=None, fig=None):
    '''
    Creates the classic Stepp (1972) plots for a completed Stepp analysis,
    and exports the figure to a file.
    :param model:
        Completed Stepp (1972) analysis as instance of :class:
        'hmtk.seismicity.completeness.comp_stepp_1971.Stepp1971'
    :param string filename:
        Name of output file
    :param string filetype:
        Type of file (from list supported by matplotlib)
    :param int filedpi:
        Resolution (dots per inch) of output file
    '''

    if (filename is not None) and os.path.exists(filename):
        raise IOError('File already exists!')

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.gca()

    # get colours from current axes: thus user can set up before calling
    valid_colours = ax._get_lines.color_cycle
    colour_cyclers = itertools.tee(itertools.cycle(valid_colours), 3)
    marker_cyclers = itertools.tee(itertools.cycle(VALID_MARKERS), 3)

    # plot observed Sigma lambda
    for i, mag_bin in enumerate(model.magnitude_bin[:-1]):
        label = '(%g, %g]: %d' % (mag_bin,
                                  mag_bin + model.config['magnitude_bin'],
                                  model.completeness_table[i, 0])
        colour = next(colour_cyclers[0])
        ax.loglog(model.time_values, model.sigma[:, i],
                  linestyle='none',
                  marker=next(marker_cyclers[0]),
                  markersize=3,
                  markerfacecolor=colour,
                  markeredgecolor=colour,
                  label=label)

    # plot expected Poisson rate
    for i in range(0, len(model.magnitude_bin) - 1):
        ax.loglog(model.time_values, model.model_line[:, i],
                  color=next(colour_cyclers[1]),
                  linewidth=0.5)

    # mark breaks from expected rate
    for i in range(0, len(model.magnitude_bin) - 1):
        colour = next(colour_cyclers[2])
        if np.any(np.isnan(model.model_line[:, i])):
            continue
        xmarker = model.end_year - model.completeness_table[i, 0]
        knee = model.model_line[:, i] > 0.
        ymarker = 10.0 ** np.interp(np.log10(xmarker),
                                    np.log10(model.time_values[knee]),
                                    np.log10(model.model_line[knee, i]))
        ax.loglog(xmarker, ymarker,
                  marker=next(marker_cyclers[2]),
                  markerfacecolor='white',
                  markeredgecolor=colour)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel("$\\sigma_{\\lambda} = \\sqrt{\\lambda} / \\sqrt{T}$")
    ax.autoscale(enable=True, axis='both', tight=True)

    # save figure to file
    if filename is not None:
        fig.savefig(filename, dpi=filedpi, format=filetype)


def plot_completeness_slices(catalogue, slice_key, slice_ids,
                             mag_bin=1., time_bin=1.,
                             deduplicate=True, mag_range=(4., None),
                             year_range=None):
    """
    Stepp (1971) analysis on sub-catalogues, where `slice_key` and
    `slice_ids` determine how the sub-catalouges are formed.
    """

    comp_config = {'magnitude_bin': mag_bin,
                   'time_bin': time_bin,
                   'increment_lock': True}

    fig, axes = plt.subplots(len(slice_ids), 1,
                             figsize=(8, 3*len(slice_ids)), sharex=True)
    fig.subplots_adjust(hspace=0)
    slice_completeness_tables = []
    for ax, slice_id in zip(axes, slice_ids):

        catalogue_slice = deepcopy(catalogue)
        in_slice = catalogue_slice.data[slice_key] == slice_id
        catalogue_slice.select_catalogue_events(in_slice)

        model = Stepp1971()
        model.completeness(catalogue_slice, comp_config)
        model.simplify(deduplicate, mag_range, year_range)
        slice_completeness_tables.append(model.completeness_table.tolist())

        plt.sca(ax)
        ax_label = '%s %d' % (slice_key, slice_id)
        ax.add_artist(AnchoredText(ax_label, loc=3, frameon=False))
        create_stepp_plot(model)

    return slice_completeness_tables
