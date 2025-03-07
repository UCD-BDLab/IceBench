# This code is authored by Andreas R. Stokholm and contributions by Tore Wulf. 
# you can find the original code here: https://github.com/astokholm/AI4ArcticSeaIceChallenge/blob/main/convert_raw_icechart.py


import copy
import numpy as np
import xarray as xr
import os

def ice_chart_mapping(scene , settings):
    """
    Original polygon_icechart in bechnmark scenes consists of codes to a lookup table `polygon_codes`.

    For SOD and FLOE the partial sea ice concentration is used to determine whether there is a dominant category in a polygon.
    The SOD and FLOE are created using the lookup tables in utils, which dictate the conversion from ice code to class, As multiple codes can be
    converted into a single class, these partial concentrations must also be added. In addition, empty codes, 'not filled values' and unknowns are
    replaced appropriately.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from the  benchmark dataset.
    """
    
    # Get codes from polygon_codes.
    codes = np.stack(np.char.split(scene['polygon_codes'].values.astype(str), sep=';'), 0)[settings['SIC_config']['total_sic_idx']:, :]
    poly_type = np.stack((codes[:, 0] , codes[:, -1]))
    codes = codes[:, :-2].astype(int) 

    # Convert codes to classes for Total and Partial SIC.
    converted_codes = copy.deepcopy(codes)
    for key, value in settings['SIC_classes'].items():
        if type(key) == int:
            for partial_idx in settings['SIC_config']['sic_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

            tmp = converted_codes[:, settings['SIC_config']['total_sic_idx']]
            if key in tmp:
                converted_codes[:, settings['SIC_config']['total_sic_idx']][np.where((tmp == key))[0]] = value

    # Find where partial concentration is empty but total SIC exist.
    ice_ct_ca_empty = np.logical_and(
        converted_codes[:, settings['SIC_config']['total_sic_idx']] > settings['SIC_classes'][0],
        converted_codes[:, settings['SIC_config']['sic_partial_idx'][0]] == settings['ICECHART_NOT_FILLED_VALUE'])
    # Assign total SIC to partial concentration when empty.
    converted_codes[:, settings['SIC_config']['sic_partial_idx'][0]][ice_ct_ca_empty] = \
            converted_codes[:, settings['SIC_config']['total_sic_idx']][ice_ct_ca_empty]

    # Convert codes to classes for partial SOD.
    for key, value in settings['SOD_classes'].items():
        if type(key) == int:
            for partial_idx in settings['SOD_config']['sod_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

    # Convert codes to classes for partial FLOE.
    for key, value in settings['FLOE_classes'].items():
        if type(key) == int:
            for partial_idx in settings['FLOE_config']['floe_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

    # Get matching partial ice classes, SOD.
    sod_a_b_bool = converted_codes[:, settings['SOD_config']['sod_partial_idx'][0]] == \
        converted_codes[:, settings['SOD_config']['sod_partial_idx'][1]]
    sod_a_c_bool = converted_codes[:, settings['SOD_config']['sod_partial_idx'][0]] == \
        converted_codes[:, settings['SOD_config']['sod_partial_idx'][2]]
    sod_b_c_bool = converted_codes[:, settings['SOD_config']['sod_partial_idx'][1]] == \
        converted_codes[:, settings['SOD_config']['sod_partial_idx'][2]]

    # Get matching partial ice classes, FLOE.
    floe_a_b_bool = converted_codes[:, settings['FLOE_config']['floe_partial_idx'][0]] == \
        converted_codes[:, settings['FLOE_config']['floe_partial_idx'][1]]
    floe_a_c_bool = converted_codes[:, settings['FLOE_config']['floe_partial_idx'][0]] == \
        converted_codes[:, settings['FLOE_config']['floe_partial_idx'][2]]
    floe_b_c_bool = converted_codes[:, settings['FLOE_config']['floe_partial_idx'][1]] == \
        converted_codes[:, settings['FLOE_config']['floe_partial_idx'][2]]

    # Remove matches where SOD == -9 and FLOE == -9.
    sod_a_b_bool[np.where(converted_codes[:, settings['SOD_config']['sod_partial_idx'][0]] == settings['ICECHART_NOT_FILLED_VALUE'])] = False
    sod_a_c_bool[np.where(converted_codes[:, settings['SOD_config']['sod_partial_idx'][0]] == settings['ICECHART_NOT_FILLED_VALUE'])] = False
    sod_b_c_bool[np.where(converted_codes[:, settings['SOD_config']['sod_partial_idx'][1]] == settings['ICECHART_NOT_FILLED_VALUE'])] = False
    floe_a_b_bool[np.where(converted_codes[:, settings['FLOE_config']['floe_partial_idx'][0]] == settings['ICECHART_NOT_FILLED_VALUE'])] = False
    floe_a_c_bool[np.where(converted_codes[:, settings['FLOE_config']['floe_partial_idx'][0]] == settings['ICECHART_NOT_FILLED_VALUE'])] = False
    floe_b_c_bool[np.where(converted_codes[:, settings['FLOE_config']['floe_partial_idx'][1]] == settings['ICECHART_NOT_FILLED_VALUE'])] = False

    # Arrays to loop over to find locations where partial SIC will be combined for SOD and FLOE.
    sod_bool_list = [sod_a_b_bool, sod_a_c_bool, sod_b_c_bool]
    floe_bool_list = [floe_a_b_bool, floe_a_c_bool, floe_b_c_bool]
    compare_indexes = [[0, 1], [0, 2], [1,2]]

    # Arrays to store how much to add to partial SIC.
    sod_partial_add = np.zeros(converted_codes.shape)
    floe_partial_add = np.zeros(converted_codes.shape)

    # Loop to find
    for idx, (compare_idx, sod_bool, floe_bool) in enumerate(zip(compare_indexes, sod_bool_list, floe_bool_list)):
        tmp_sod_bool_indexes = np.where(sod_bool)[0]
        tmp_floe_bool_indexes = np.where(floe_bool)[0]
        if tmp_sod_bool_indexes.size:  #i.e. is array is not empty.
            sod_partial_add[tmp_sod_bool_indexes, settings['SIC_config']['sic_partial_idx'][compare_idx[0]]] = \
                converted_codes[:, settings['SIC_config']['sic_partial_idx'][compare_idx[1]]][tmp_sod_bool_indexes]

        if tmp_floe_bool_indexes.size:  # i.e. is array is not empty.
            floe_partial_add[tmp_floe_bool_indexes, settings['SIC_config']['sic_partial_idx'][compare_idx[0]]] = \
                converted_codes[:, settings['SIC_config']['sic_partial_idx'][compare_idx[1]]][tmp_floe_bool_indexes]

    # Create arrays for charts.
    scene_tmp = copy.deepcopy(scene['polygon_icechart'].values)
    sic = copy.deepcopy(scene['polygon_icechart'].values)
    sod = copy.deepcopy(scene['polygon_icechart'].values)
    floe = copy.deepcopy(scene['polygon_icechart'].values)

    # Add partial concentrations when classes have been merged in conversion (see SIC, SOD, FLOE tables).
    tmp_sod_added = converted_codes + sod_partial_add.astype(int)
    tmp_floe_added = converted_codes + floe_partial_add.astype(int)

    # Find and replace all codes with SIC, SOD and FLOE.
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(codes.shape[0]):
            code_match = np.where(scene_tmp == converted_codes[i, settings['polygon_idx']])
            sic[code_match] = converted_codes[i, settings['SIC_config']['total_sic_idx']]

            if np.char.lower(poly_type[1, i]) == 'w':
                sic[code_match] = settings['SIC_classes'][0]
            
            # Check if there is a class combined normalized partial concentration, which is dominant in the polygon.
            if np.divide(np.max(tmp_sod_added[i, settings['SIC_config']['sic_partial_idx']]),
                    tmp_sod_added[i, settings['SIC_config']['total_sic_idx']]) * 100 >= settings['SOD_config']['threshold'] * 100:

                # Find dominant partial ice type.
                sod[code_match] = converted_codes[i, settings['SOD_config']['sod_partial_idx']][
                    np.argmax(tmp_sod_added[i, settings['SIC_config']['sic_partial_idx']])]
            else:
                sod[code_match] = settings['ICECHART_NOT_FILLED_VALUE']
            
            # Check if there is a class combined normalized partial concentration, which is dominant in the polygon.
            if np.divide(np.max(tmp_floe_added[i, settings['SIC_config']['sic_partial_idx']]),
                    tmp_floe_added[i, settings['SIC_config']['total_sic_idx']]) * 100 >= settings['FLOE_config']['threshold'] * 100:
                floe[code_match] = converted_codes[i, settings['FLOE_config']['floe_partial_idx']][
                    np.argmax(tmp_floe_added[i, settings['SIC_config']['sic_partial_idx']])]
            else:
                floe[code_match] = settings['ICECHART_NOT_FILLED_VALUE']

            if any(converted_codes[i, settings['FLOE_config']['floe_partial_idx']] == settings['FLOE_config']['fastice_class']):
                floe[code_match] = settings['FLOE_config']['fastice_class']

    # Add masked pixels for ambiguous polygons.
    sod[sod == settings['SOD_config']['invalid']] = settings['SOD_config']['mask']
    floe[floe == settings['FLOE_config']['invalid']] = settings['FLOE_config']['mask']

    # Ensure water is identical across charts.
    sod[sic == settings['SIC_classes'][0]] = settings['SOD_config']['water']
    floe[sic == settings['SIC_classes'][0]] = settings['FLOE_config']['water']

    # Add the new charts to scene and add descriptions:
    scene = scene.assign({'SIC': xr.DataArray(sic, dims=scene['polygon_icechart'].dims)})
    scene = scene.assign({'SOD': xr.DataArray(sod, dims=scene['polygon_icechart'].dims)})
    scene = scene.assign({'FLOE': xr.DataArray(floe, dims=scene['polygon_icechart'].dims)})
    
    for chart in settings['charts']:
        # Remove any unknowns.
        scene[chart].values[scene[chart].values == int(settings['ICECHART_UNKNOWN'])] = settings['LOOKUP_NAMES'][chart]['mask']
        # remove the nan values
        scene[chart].values[np.isnan(scene[chart].values)] = settings['LOOKUP_NAMES'][chart]['mask']
        
        scene[chart].attrs = ({
            'polygon': settings['ICE_STRINGS'][chart],
            'chart_fill_value': settings['LOOKUP_NAMES'][chart]['mask']
        })

    return scene
