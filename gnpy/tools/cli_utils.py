#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# Utils functions for command line interface script
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.tools.cli_utils
=======================

Utils for creation of export result
"""

from pathlib import Path
import logging
import re
from csv import writer
from copy import deepcopy
from typing import Union, List, Optional, Dict
from math import ceil
import xlsxwriter
from numpy import mean
from gnpy.core import elements
from gnpy.core.utils import pretty_summary_print, per_label_average, watt2dbm
from gnpy.topology.spectrum_assignment import mvalue_to_slots, nvalue_to_frequency, BitmapValue, OMS
from gnpy.topology.request import PathRequest, ResultElementTrans, ResultElement, jsontocsv
from gnpy.tools.json_io import save_gnpy_json

_logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = ['.json', '.csv', '']


def oms_summary(oms_list: List, requests: List[PathRequest], export_all_lines: bool = True) \
        -> List[List[Union[str, int, float]]]:
    """Prepare the summary of all oms spectral occupation including
    their remaining capability and fragmentation level.

    :param oms_list: A list of OMS objects.
    :type oms_list: List
    :param requests: A list of PathRequest objects.
    :type requests: List[PathRequest]
    :param export_all_lines: If False, only export the lines that contain at least a service, else
                             export all of them.
    :type export_all_lines: bool
    :return: A summary list containing details about each OMS.
    :rtype: List[List[Union[str, int, float]]]
    """
    summary = [['oms_id', 'free_slots', 'occupied_slots', 'nb of channels', 'remaining 5OG channels',
                'remaining 75G channels', 'remaining 100G channels', 'ABP', 'useable capacity (Gbit/s)',
                ' spectrum efficiency', 'carried services']]
    explored_oms = []
    req_dict = {r.request_id: r.bit_rate * ceil(r.path_bandwidth / r.bit_rate) if r.bit_rate else 0 for r in requests}
    for oms in oms_list:
        # Show the nb of occupied/free slots, which requests are going through it,
        # how many 50G, 75G or 100G it can fit, eventually some fragmentation/health metric
        export_line = export_all_lines or oms.service_list
        if oms.oms_id not in explored_oms and export_line:
            bitmap = oms.spectrum_bitmap.bitmap
            nb_free = sum(1 for b in bitmap if b == BitmapValue.FREE)
            nb_occupied = sum(1 for b in bitmap if b == BitmapValue.OCCUPIED)
            nb_50 = count_useable_spectrum(bitmap, 50e9)
            nb_75 = count_useable_spectrum(bitmap, 75e9)
            nb_100 = count_useable_spectrum(bitmap, 100e9)
            abp = access_blocking_probability(list_continuous_free_slot(bitmap), [50e9, 75e9, 100e9])
            useable_capacity = 0
            for s in oms.service_list:
                useable_capacity += req_dict[s]
            spectrum_efficiency = useable_capacity * 1e-9 / (nb_occupied * 6.25) if useable_capacity > 0 else None
            line = [ietf_oms_name(oms), nb_free, nb_occupied, oms.nb_channels, nb_50, nb_75, nb_100, abp,
                    useable_capacity * 1e-9, spectrum_efficiency, oms.service_list]
            summary.append(line)
            if oms.reversed_oms:
                explored_oms.append(oms.reversed_oms.oms_id)
    return summary


def save_table_in_csv(my_table: List[List[Union[str, int, float]]], filename: str):
    """Records oms spectral info into a csv file.

    :param my_table: A table of data to save.
    :type my_table: List[List[Union[str, int, float]]]
    :param filename: The name of the file to save the data to.
    :type filename: str
    """
    msg = f'Preparing summary to be saved in: {filename}'
    print(msg)
    _logger.info(msg)
    with open(filename, "w", encoding='utf-8') as fileout:
        mywriter = writer(fileout)
        for elem in my_table:
            mywriter.writerow(elem)


def range_n(one_n: int, one_m: int) -> range:
    """ transform [a, b] into a, a+1, a+2, ... b

    :param one_n: The start of the range.
    :type one_n: int
    :param one_m: The end of the range.
    :type one_m: int
    :return: A range object representing the integers from one_n to one_m.
    :rtype: range
    """
    start, stop = mvalue_to_slots(one_n, one_m)
    return range(start, stop + 1, 1)


def range_list(req_n: List[int], req_m: List[int]) -> List[int]:
    """Transform list of (n, m), ito a single index list.

    :param req_n: A list of starting values.
    :type req_n: List[int]
    :param req_m: A list of ending values.
    :type req_m: List[int]
    :return: A list of integers representing the combined ranges.
    :rtype: List[int]

    >>> range_list([0, 20], [4, 4])
    [-4, -3, -2, -1, 0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23]
    """
    n_list = []
    for n, m in zip(req_n, req_m):
        n_list.extend(range_n(n, m))
    return n_list


def ietf_oms_name(oms) -> str:
    """Use the link-id used in element prefix
    regex only applicable to toaster: suppose that first element contains the OMS prefix
    else uses roadm-roadm label
    TODO: to be changed when parallel links will be handled

    :param oms: The OMS object to get the name from.
    :type oms: Any
    :return: A string representing the IETF OMS name.
    :rtype: str
    """
    my_regex = re.compile(r'\[[0-9A-Za-z/\-=>]*\]')
    if my_regex.findall(oms.el_id_list[1]):
        # case of IETF naming of the element link-id as prefix
        return f'{oms.el_id_list[1].split("]")[0]}]'
    return f'{oms.el_id_list[0]} - {oms.el_id_list[-1]}'


def oms_spectral_graph(oms_list: List, rqs: List[PathRequest], export_all_lines: bool = True) \
        -> List[List[Union[str, float]]]:
    """Prepare a table with services id instead of 0 in each oms bitmap.

    :param oms_list: A list of OMS objects.
    :type oms_list: List
    :param rqs: A list of PathRequest objects.
    :type rqs: List[PathRequest]
    :param export_all_lines: If False, only export the lines that contain at least a service, else
                             export all of them.
    :type export_all_lines: bool
    :return: A table representing the spectral graph for each OMS.
    :rtype: List[List[Union[str, float]]]
    """
    services = {r.request_id: range_list(r.N, r.M) for r in rqs if r.N is not None and r.M is not None}
    line0 = ['oms_id', '6.25GHz slots occupations']
    # retrieve bitmap borders within oms list
    n_min = oms_list[0].spectrum_bitmap.n_min
    n_max = oms_list[0].spectrum_bitmap.n_max
    line0.extend(list(range(n_min, n_max + 1)))
    line1 = ['', '']
    line1.extend([round(nvalue_to_frequency(n) * 1e-12, 5) for n in range(n_min, n_max + 1)])
    occupation_table = [line0, line1]
    explored_oms = []
    errors = {}
    for oms in oms_list:
        export_line = export_all_lines or oms.service_list
        if oms.oms_id not in explored_oms and export_line:
            oms_bitmap = deepcopy(oms.spectrum_bitmap)
            for service in oms.service_list:
                for n_value in services[service]:
                    if oms_bitmap.bitmap[oms_bitmap.geti(n_value)] == BitmapValue.OCCUPIED:
                        oms_bitmap.bitmap[oms_bitmap.geti(n_value)] = service
                    else:
                        if ietf_oms_name(oms) not in errors:
                            errors[ietf_oms_name(oms)] = {}
                        if ietf_oms_name(oms) in errors and service not in errors[ietf_oms_name(oms)]:
                            errors[ietf_oms_name(oms)][service] = []
                        # collect all errors before stopping
                        errors[ietf_oms_name(oms)][service].append(n_value)
            # change BitmapValue.FREE values to empty for the printing except if it is a service name!
            for i, bt in enumerate(oms_bitmap.bitmap):
                if bt == BitmapValue.FREE:
                    oms_bitmap.bitmap[i] = ''
                elif bt == BitmapValue.UNUSABLE:
                    oms_bitmap.bitmap[i] = 'u'
            line = [ietf_oms_name(oms), ''] + oms_bitmap.bitmap
            occupation_table.append(line)
            if oms.reversed_oms:
                explored_oms.append(oms.reversed_oms.oms_id)
    if errors:
        msg = 'oms_spectral_graph export: inconsistency between services spectrum and oms spectrum bitmap\n'
        for oms_name, item in errors.items():
            for service_name, bits in item.items():
                msg += f'OMS {oms_name}: {service_name} not consistant with {bits}\n'
        _logger.error(msg)
    return occupation_table


def list_continuous_free_slot(bitmap: List[BitmapValue]) -> List[int]:
    """List each sub bitmap with contiguous spectrum.

    :param bitmap: A list representing the bitmap of the spectrum.
    :type bitmap: List[BitmapValue]
    :return: A list of integers representing the sizes of contiguous free slots.
    :rtype: List[int]
    """
    list_free_slots = []
    i = 0
    while i < len(bitmap):
        nb = 0
        if bitmap[i] == BitmapValue.FREE:
            while i + nb + 1 < len(bitmap) and bitmap[i + nb + 1] == BitmapValue.FREE:
                nb += 1
            list_free_slots.append(nb)
            i = i + nb + 1
        i = i + 1
    return list_free_slots


def count_useable_spectrum(bitmap: List[BitmapValue], spacing: float) -> int:
    """Count the number of channels one can put in a bitmap.

    :param bitmap: A list representing the bitmap of the spectrum.
    :type bitmap: List[BitmapValue]
    :param spacing: The spacing in Hz for the channels.
    :type spacing: float
    :return: The number of usable channels.
    :rtype: int
    """
    contiguous_free_slots = list_continuous_free_slot(bitmap)
    nb = 0
    for slots in contiguous_free_slots:
        nb += slots // (spacing / 6.25e9)
    return nb


def access_blocking_probability(slots: List[int], spacings: List[float]) -> float:
    """Computes the ABP for a given set of slots.

    :param slots: A list of integers representing the number of free slots.
    :type slots: List[int]
    :param spacings: A list of floats representing the spacing values.
    :type spacings: List[float]
    :return: The calculated ABP.
    :rtype: float
    """
    granularities = [s // 6.25e9 for s in spacings]
    numerator = 0
    for elem in slots:
        for gran in granularities:
            numerator += elem // gran
    denominator = 0
    nb_slots = sum(slots)
    for gran in granularities:
        denominator += nb_slots // gran
    return 1 - numerator / denominator


COLORS = ['#0000FF', '#800000', '#00FFFF', '#008000', '#00FF00', '#FF00FF', '#000080',
          '#FF6600', '#FF00FF', '#800080', '#FF0000', '#C0C0C0', '#FFFF00']


def write_xls(ots_summary: List[List[Union[str, int, float]]], occupation_table: List[List[Union[str, float]]],
              output_filename: str):
    """Printing the summaries of spectral occupation.

    - one sheet with summary per OMS, recording available and unavailable number of slots, fragmentation metric,
      average spectral efficiency (bit rate / occupied spectrum),
    - one sheet showing the spectral grid for each OMS with each service occupation.

    :param ots_summary: A summary of the optical transport system.
    :type ots_summary: List[List[Union[str, float]]]
    :param occupation_table: A table showing the spectral occupation.
    :type occupation_table: List[List[Union[str, float]]]
    :param output_filename: The name of the file to save the data to.
    :type output_filename: str
    """
    msg = f'Preparing oms + spectrum summary to be saved in: {output_filename}'
    print(msg)
    _logger.info(msg)
    workbook = xlsxwriter.Workbook(output_filename, {'constant_memory': True})
    sheet = workbook.add_worksheet('ots_summary')

    for line, ots in enumerate(ots_summary):
        for column, value in enumerate(ots):
            if isinstance(value, list):
                for i, elem in enumerate(value):
                    sheet.write(line, column + i, elem)
            else:
                sheet.write(line, column, value)

    cell_format = workbook.add_format()
    cell_format.set_text_wrap()
    sheet = workbook.add_worksheet('occupation_table')
    # set_column(first_col, last_col, width, cell_format, options)
    sheet.set_column(0, 0, 30)
    sheet.set_column(1, 2000, 2)
    for line, ots in enumerate(occupation_table):
        sheet.set_row(line, 30)
        val = 0
        nb = 0
        color = 'white'
        for column, value in enumerate(ots):
            if line > 1 and column == 0:
                sheet.write(line, column, value, cell_format)
            elif line in [0, 1]:
                sheet.set_row(line, 80)
                header_format = workbook.add_format()
                header_format.set_bg_color('gray')
                header_format.set_rotation(90)
                sheet.write(line, column, value, header_format)
            else:
                line_format = workbook.add_format()
                line_format.set_rotation(90)
                if value not in (val, '', 'u'):
                    # first time this service id is met, use a new color, record it in val
                    nb += 1
                    color = COLORS[nb % len(COLORS)]
                    val = value
                    line_format.set_bg_color(color)
                elif val == value and value != '' and value != 'u':
                    # use the same color for this service
                    line_format.set_bg_color(color)
                elif value == '' and value != 'u':
                    # do not write anything
                    continue
                elif value == 'u':
                    # use a dark grey to color unuseable bits
                    line_format.set_bg_color('#9D9D9D')
                sheet.write(line, column, value, line_format)
    workbook.close()
    msg = f'Saved oms + spectrum summary in: {output_filename}'
    print(msg)
    _logger.info(msg)


def write_path_xls(path_array: List[str], output_filename: str):
    """Path array contains the path details settings and powers.

    :param path_array: A list of path details.
    :type path_array: List[str]
    :param output_filename: The name of the file to save the path details to.
    :type output_filename: str
    """
    msg = f'Preparing path array to be saved in: {output_filename}'
    print(msg)
    _logger.info(msg)
    workbook = xlsxwriter.Workbook(output_filename, {'constant_memory': True})
    sheet = workbook.add_worksheet('path detail')
    cell_format = workbook.add_format()
    cell_format.set_text_wrap()
    sheet.set_column(0, 100, 30)
    sheet.set_default_row(20)
    width = {}
    for line, path in enumerate(path_array):
        for column, value in enumerate(path):
            width[line] = max(width[line], 18 * (value.count('\n') + 1)) if line in width else 18
            sheet.set_row(line, width[line], cell_format)
            sheet.write(line, column, value, cell_format)

    workbook.close()
    msg = f'Saving path array in: {output_filename}'
    print(msg)
    _logger.info(msg)


def prepare_detailed_path(propagatedpths: List[List[Union[elements.Transceiver, elements.Fiber, elements.RamanFiber,
                                                          elements.Edfa, elements.Multiband_amplifier,
                                                          elements.Fused]]],
                          rqs: List[PathRequest]) -> List[str]:
    """Print detailed path prepared by PINT.

    :param propagatedpths: A list of paths to be printed.
    :type propagatedpths: List[List[Union[elements.Transceive, elements.Fiber, elements.RamanFiber, elements.Edfa,
                          elements.Multiband_amplifier, elements.Fused]]]
    :param rqs: A list of PathRequest objects.
    :type rqs: List[PathRequest]
    :return: A list of strings representing the detailed paths.
    :rtype: List[str]
    """
    path_array = []
    for i, p in enumerate(propagatedpths):
        if -1 in range(-len(p), len(p)):

            path_array.append([f'request {rqs[i].request_id}'])
            path_array.append([f'{rqs[i].source} to {rqs[i].destination}'])
            path_array.append([f'snr@0.1nm : {round(mean(p[-1].snr_01nm), 2)} dB',   # noqa E203
                               f'Receiver minOSNR : {rqs[i].OSNR} dB'])   # noqa E203
            chemin = []
            for var_element in p:
                class_name = type(var_element).__name__
                obj_name = var_element.name
                if class_name in 'Transceiver':
                    chemin.append(f'{class_name} {str(obj_name)}' + transceiver_string(var_element))
                elif class_name == 'Roadm':
                    chemin.append(f'{class_name} {str(obj_name)}' + roadm_string(var_element))
                elif class_name in ['Fiber', 'RamanFiber']:
                    chemin.append(f'{class_name} {str(obj_name)}' + fiber_string(var_element))
                elif class_name == 'Edfa':
                    chemin.append(f'{class_name} {str(obj_name)}' + amp_string(var_element))
                elif class_name == 'Multiband_amplifier':
                    multiband_amp = '\n'
                    for amp in var_element.amplifiers.values():
                        band = f'\tband {amp.params.f_min * 1e-12:3.3f}-{amp.params.f_max * 1e-12:3.3f} THz'  # noqa E231 # pylint: disable=C0301
                        multiband_amp = multiband_amp + band + amp_string(amp, tab='\n\t\to')
                    chemin.append(f'{class_name} {str(obj_name)}' + multiband_amp)
                elif class_name == 'Fused':
                    chemin.append(f'{class_name} {str(obj_name)}' + fused_string(var_element))
                else:
                    raise ValueError('Unrecognized class type ', class_name)
            path_array.append(chemin)
    return path_array
    # PINT


def line_str(message: str, unit: str, variable: Union[str, float, int], rounding: Union[int, None],
             tab: str = '\n\t-') -> str:
    """Build the bullet point string encompassing None case.

    :param message: The message to display.
    :type message: str
    :param unit: The unit of measurement.
    :type unit: str
    :param variable: The value to display.
    :type variable: Union[str, float, int]
    :param rounding: The number of decimal places to round to (if applicable).
    :type rounding: Union[int, None]
    :return: A formatted string representing the line.
    :rtype: str
    """
    if variable and rounding:
        return f'{tab} {message}: {round(variable, rounding)} {unit}'
    if variable and unit:
        return f'{tab} {message}: {variable} {unit}'
    if variable and not unit:
        return f'{tab} {message}: {variable}'
    if not variable and unit:
        return f'{tab} {message}: {variable} {unit}'
    return f'{tab} {message}: {variable}'


def amp_string(element: elements.Edfa, tab: str = '\n\t-') -> str:
    """Build the amplifier type string.

    :param element: The Edfa object representing the amplifier.
    :type element: elements.Edfa
    :return: A string representing the amplifier details.
    :rtype: str
    """
    nf_values = []
    if hasattr(element, "pch_out_dbm"):
        total_pch = pretty_summary_print(per_label_average(element.pch_out_dbm, element.propagated_labels))
    else:
        total_pch = None
    if hasattr(element, "nf") and element.nf is not None:
        nf_values = [element.nf]
    else:
        nf_values = []
    nf_mean = round(mean(nf_values), 2) if nf_values else None
    nf = nf_mean if nf_mean is not None else None
    return line_str('Type', '', element.params.type_variety, None, tab) +\
        line_str('Effective_gain (before att_in and before output VOA)', 'dB', element.effective_gain, 3, tab) +\
        line_str('P in', 'dBm', element.pin_db, 3, tab) +\
        line_str('P out', 'dBm', element.pout_db, 3, tab) +\
        line_str('P out/ch', 'dBm', total_pch, None, tab) +\
        line_str('NF', 'dB', nf, None, tab) +\
        line_str('Out VOA', 'dB', element.out_voa, 2, tab) +\
        line_str('In VOA', 'dB', element.in_voa, 2, tab) +\
        line_str('Att in', 'dB', element.att_in, 2, tab) + '\n'


def fiber_string(element: Union[elements.Fiber, elements.RamanFiber]) -> str:
    """Build the fiber type string.

    :param element: The Fiber or RamanFiber object representing the fiber.
    :type element: Union[elements.Fiber, elements.RamanFiber]
    :return: A string representing the fiber details.
    :rtype: str
    """
    return line_str('Type', '', element.type_variety, None) +\
        line_str('Total loss (totale inc. conn loss and padding)', 'dB', element.loss, 3) +\
        line_str('Length', 'km', element.params.length / 1000, 3) +\
        line_str('Loss coefficient', 'dB/km', element.params.loss_coef * 1e3, 4) +\
        line_str('Conn in loss', 'dB', element.params.con_in, None) +\
        line_str('Conn out loss', 'dB', element.params.con_out, None) +\
        line_str('Padding in', 'dB', element.params.att_in, None) +\
        line_str('Chromatic dispersion', 's/m', mean(element.chromatic_dispersion()), 6) + '\n'


def fused_string(element: elements.Fused) -> str:
    """Build the fused element type string.

    :param element: The Fused object representing the fused element.
    :type element: elements.Fused
    :return: A string representing the fused element details.
    :rtype: str
    """
    return line_str('Loss', 'dB', element.loss, 3) + '\n'


def roadm_string(element: elements.Roadm) -> str:
    """Build the ROADM type string.

    :param element: The Roadm object representing the ROADM.
    :type element: elements.Roadm
    :return: A string representing the ROADM details.
    :rtype: str
    """
    total_loss = pretty_summary_print(per_label_average(element.loss_pch_db, element.propagated_labels))
    return line_str('Actual loss', 'dB', total_loss, None) + '\n'


def transceiver_string(element: elements.Transceiver) -> str:
    """Build the transceiver type string.

    :param element: The transceiver object representing the transceiver.
    :type element: elements.Transceive
    :return: A string representing the transceiver details.
    :rtype: str
    """
    if element.penalties:
        cd = mean(element.penalties.get('chromatic_dispersion')) \
            if element.penalties.get('chromatic_dispersion') is not None else None
        pmd = mean(element.penalties.get('pmd')) if element.penalties.get('pmd') is not None else None
        pdl = mean(element.penalties.get('pdl')) if element.penalties.get('pdl') is not None else None
        penalties_string = line_str('CD penalty', 'dB', cd, 2) +\
            line_str('PMD penalty:', 'dB', pmd, 2) +\
            line_str('PDL penalty:', 'dB', pdl, 3) + '\n'
    else:
        penalties_string = '\n'

    return line_str('GSNR(0.1nm):', 'dB', pretty_summary_print(
        (per_label_average(element.snr_01nm, element.propagated_labels))), None) \
        + line_str('GSNR(signal bw):', 'dB', pretty_summary_print(
            (per_label_average(element.snr, element.propagated_labels))), None) \
        + line_str('OSNR ASE (0.1nm):', 'dB', pretty_summary_print(
            (per_label_average(element.osnr_ase_01nm, element.propagated_labels))), None) \
        + line_str('OSNR ASE(signal bw):', 'dB', pretty_summary_print(
            (per_label_average(element.osnr_ase, element.propagated_labels))), None) \
        + line_str('CD:', 'ps/nm', pretty_summary_print(
            (per_label_average(element.chromatic_dispersion, element.propagated_labels))), None) \
        + line_str('PMD:', 'ps', pretty_summary_print(
            (per_label_average(element.pmd, element.propagated_labels))), None) \
        + line_str('PDL:', 'dB', pretty_summary_print(
            (per_label_average(element.pdl, element.propagated_labels))), None) \
        + line_str('Latency', 'ms', mean(element.latency), 2) \
        + line_str('Actual pch out:', 'dBm', pretty_summary_print(
            (per_label_average(watt2dbm(element.tx_power), element.propagated_labels))), None) \
        + penalties_string  # noqa E501   # pylint: disable=line-too-long


def save_path_array_to_xlsx(output_path: Path,
                            propagations_for_path: List[List[Union[elements.Transceiver, elements.Fiber,
                                                                   elements.RamanFiber,
                                                                   elements.Edfa, elements.Multiband_amplifier,
                                                                   elements.Fused]]],
                            req: List[PathRequest]) -> Path:
    """Save results in xlsx format

    :param output_path: Path of the output file specified by the user
    :type output_path: Path
    :param propagations_for_path: A list of lists containing objects of type
                                  Transceiver, Fiber, RamanFiber, Edfa,
                                  Multiband_amplifier, or Fused.
    :type propagations_for_path: List[List[Union[elements.Transceiver, elements.Fiber, elements.RamanFiber,
                                 elements.Edfa, elements.Multiband_amplifier, elements.Fused]]]
    :param req: Object containing request information
    :type req:PathRequest
    :return: Path of the created xlsx file.
    :rtype: Path
    """

    output_filename = output_path.with_suffix('.xlsx')
    msg = f'Preparing path array to be saved in: {output_filename}'
    print(msg)
    _logger.info(msg)
    path_array = prepare_detailed_path(propagations_for_path, req)
    write_path_xls(path_array, output_filename)
    return output_filename


def _path_result_json(pathresult: List):
    """
    Create the result dictionary (response for a request).

    :param pathresult: List of objects, each having a 'json' attribute.
    :type pathresult: list

    :return: A dictionary with a 'response' key containing a list of JSON representations.
    :rtype: dict
    """
    return {'response': [n.json for n in pathresult]}


def generate_final_json(rqs: List[PathRequest],
                        propagatedpths: List[List[Union[elements.Transceiver, elements.Fiber, elements.RamanFiber,
                                                        elements.Edfa, elements.Multiband_amplifier,
                                                        elements.Fused]]]):
    """
    Generates a JSON response for path computation based on requests and propagated paths.

    :param rqs: List of PathRequest objects, each containing a request_id.
    :type rqs: list of PathRequest

    :param propagatedpths: List of propagated path elements, each being a list of equipment objects
                         such as Transceiver, Fiber, RamanFiber, Edfa, Multiband_amplifier, or Fused.
    :type propagatedpths: list of list of Union[elements.Transceiver, elements.Fiber, elements.RamanFiber,
                          elements.Edfa, elements.Multiband_amplifier, elements.Fused]

    :return: A dictionary representing the JSON response with path properties and route objects.
    :rtype: dict
    """
    path_id = rqs[0].request_id

    last_element = propagatedpths[0][-1]
    result_element = ResultElementTrans(rqs[0], last_element, len(propagatedpths[0]) - 1)

    path_properties = {
        "path-metric": result_element.path_properties
    }

    path_route_objects = []
    for index, element in enumerate(propagatedpths[0]):
        result_element = ResultElementTrans(rqs[0], element, index)
        path_route_objects.append({"path-route-object": result_element.pathresult["path-route-objects"]})

    path_properties["path-route-objects"] = path_route_objects

    final_json = {
        "gnpy-path-computation:responses": {
            "response": [
                {
                    "response-id": path_id,
                    "path-properties": path_properties
                }
            ]
        }
    }

    return final_json


def save_json_path_and_trans(output_path: Path,
                             propagatedpths: List[List[Union[elements.Transceiver, elements.Fiber, elements.RamanFiber,
                                                             elements.Edfa, elements.Multiband_amplifier,
                                                             elements.Fused]]],
                             rqs: List[PathRequest], equipment: Dict, flag: str, oms_list: Optional[List[OMS]],
                             result: Optional[List[ResultElement]]):
    """
    Saves the path and transmission results in various formats based on the specified flag.

    :param output_path: The output file path where results will be saved.
    :type output_path: Path
    :param propagatedpths: A list of lists containing propagated path elements (transceivers, fibers, etc.).
    :type propagatedpths: Union[elements.Transceiver, elements.Fiber, elements.RamanFiber,
                          elements.Edfa, elements.Multiband_amplifier, elements.Fused]
    :param rqs: List of PathRequest objects representing the requests.
    :type rqs: list of PathRequest
    :param equipment: Equipment data used for CSV and XLS exports.
    :type equipment: Dict
    :param flag: A string indicating the type of output ('Path' or 'Trans').
    :type flag: str
    :param oms_list: List of OMS data used for summaries and spectra.
    :type oms_list: List[OMS]
    :return: None. Results are saved to files based on the provided path and flag.
    :raises ValueError: If the output format is unsupported or flag is unknown.
    """
    # pylint: disable R0915
    prefix = str(output_path)[0:len(str(output_path)) - len(str(output_path.suffix))]
    if flag == "Path":
        temp = _path_result_json(result)

        # csv and json files saving
        if output_path.suffix.lower() in SUPPORTED_SUFFIXES:
            save_gnpy_json(temp, f'{prefix}.json')
            msg = f'Saved JSON to {f"{prefix}.json"}'
            print(msg)
            _logger.info(msg)
            with open(f'{prefix}.csv', "w", encoding='utf-8') as fcsv:
                jsontocsv(temp, equipment, fcsv)
            msg = f'Saved CSV to {f"{prefix}.csv"}'
            print(msg)
            _logger.info(msg)
        else:
            msg = 'Cannot save output: neither JSON nor CSV file'
            print(msg)
            _logger.error(msg)
            raise ValueError(msg)

        # detailled path
        path_array = prepare_detailed_path(propagatedpths, rqs)
        fnamexls = f'{str(output_path)[0:len(str(output_path)) - len(str(output_path.suffix))]}_PATH.xlsx'
        write_path_xls(path_array, fnamexls)

        # oms summary
        oms_summary_name = f'{prefix}_oms.csv'
        oms_summ = oms_summary(oms_list, rqs, export_all_lines=True)
        save_table_in_csv(oms_summ, oms_summary_name)

        # spectrum and occupation
        oms_graph_name = f'{prefix}_spectrumgraph.csv'
        occupation_table = oms_spectral_graph(oms_list, rqs, export_all_lines=True)
        save_table_in_csv(occupation_table, oms_graph_name)

        # excel file
        oms_name = f'{prefix}_oms_spectrumgraph.xlsx'
        write_xls(oms_summ, occupation_table, oms_name)

    elif flag == "Trans":
        final_json = generate_final_json(rqs, propagatedpths)

        if output_path.suffix.lower() in ['.json', '.xlsx']:
            save_gnpy_json(final_json, f'{prefix}.json')
            msg = f'Saved JSON to {f"{prefix}.json"}'
            print(msg)
            _logger.info(msg)
            result_file = save_path_array_to_xlsx(output_path, propagatedpths, rqs)
            msg = f'\nResults saved in {result_file}'
            print(msg)
            _logger.info(msg)
        else:
            msg = f'Cannot save output in {output_path}: neither JSON nor CSV file'
            _logger.error(msg)
            raise ValueError(msg)
    else:
        msg = "Flag unkown or missing. use 'Path' or 'Trans'."
        _logger.error(msg)
        raise ValueError(msg)
