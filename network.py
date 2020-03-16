"""Definition of the network model and various RNN cells"""

from __future__ import division

import os
import numpy as np
import pickle
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell

import tools


def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    num_units = y.get_shape().as_list()[-1]
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / num_units)  # preferences
    cos_pref = np.cos(pref)
    sin_pref = np.sin(pref)
    temp_sum = tf.reduce_sum(input_tensor=y, axis=-1)
    temp_cos = tf.reduce_sum(input_tensor=y * cos_pref, axis=-1) / temp_sum
    temp_sin = tf.reduce_sum(input_tensor=y * sin_pref, axis=-1) / temp_sum
    loc = tf.atan2(temp_sin, temp_cos)
    return tf.math.mod(loc, 2*np.pi)


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf


def all_network_architectures(hp):
    n_exc_units_module = int(int(hp['n_rnn'] / 5) * hp['exc_prop_RNN'])
    n_inh_units_module = int(hp['n_rnn'] / 5) - n_exc_units_module
    if 'inh_prop_TRN' in hp:
        n_TRN_units = int(int(hp['n_rnn'] / 5) * hp['inh_prop_TRN'])
        n_exc_thal_units = int(hp['n_rnn'] / 5) - n_TRN_units
    else:
        n_TRN_units = n_inh_units_module
        n_exc_thal_units = n_exc_units_module

    network_architectures = {
        'basic_TC': { # multiple cortical modules
            'sen_input': {
                'n_modules': 3,
                'pre': [range(1 + i * hp['n_eachring'], 1 + (i+1) * hp['n_eachring']) if i >= 0 else [0] for i in range(-1,2)],
                'post': [range(i * int(hp['n_rnn'] / 5), (i+1) * int(hp['n_rnn'] / 5)) for i in range(3)],
                'rec': [False for i in range(3)],
                'EI_balance': [False for i in range(3)],
                'exc_prop': [None for i in range(3)]},
            'rule_input': {
                'n_modules': 1,
                'pre': ['all'],
                'post': [range(0, int(hp['n_rnn']/5))],
                'rec': [False],
                'EI_balance': [False],
                'exc_prop': [None]},
            'rnn': {
                'n_modules': 5,
                'pre': [range(i * int(hp['n_rnn'] / 5), (i+1) * int(hp['n_rnn'] / 5)) if i < 4 else range(4 * int(hp['n_rnn'] / 5), hp['n_rnn']) for i in range(5)],
                'post': [range(4 * int(hp['n_rnn'] / 5), hp['n_rnn']) if i < 4 else range(0, 4 * int(hp['n_rnn'] / 5)) for i in range(5)],
                'rec': [True if i < 4 else False for i in range(5)],
                'EI_balance': [False for i in range(5)],
                'exc_prop': [None for i in range(5)]},
            'output': {
                'n_modules': 1,
                'pre': ['all'],
                'post': [range(3 * int(hp['n_rnn'] / 5), 4 * int(hp['n_rnn'] / 5))],
                'rec': [False],
                'EI_balance': [False],
                'exc_prop': [None]}
        },
        'single_module_TC_with_TRN_v2': {
            'sen_input': {
                'n_modules': 1,
                'pre': ['all'],
                'post': [range(int(0.85 * hp['n_rnn']), int(0.9 * hp['n_rnn']))], # to FO thalamus
                'rec': [False],
                'EI_balance': [False],
                'exc_prop': [None]},
            'rule_input': {
                'n_modules': 1,
                'pre': ['all'],
                'post': [range(0, int(0.75 * hp['n_rnn']))], # to cortex
                'rec': [False],
                'EI_balance': [False],
                'exc_prop': [None]},
            'rnn': {
                'n_modules': 6,
                'pre': [range(0, int(0.75 * hp['n_rnn'] * (1 - hp['exc_prop_RNN']))), # from inh cortex
                        range(int(0.75 * hp['n_rnn'] * (1 - hp['exc_prop_RNN'])), int(0.75 * hp['n_rnn'] * (1 - hp['exc_prop_RNN']/2))), # from CC cortex
                        range(int(0.75 * hp['n_rnn'] * (1 - hp['exc_prop_RNN']/2)), int(0.75 * hp['n_rnn'])), # from CT cortex
                        range(int(0.75 * hp['n_rnn']), int(0.85 * hp['n_rnn'])), # from TRN
                        range(int(0.85 * hp['n_rnn']), int(0.9 * hp['n_rnn'])), # from FO thalamus
                        range(int(0.9 * hp['n_rnn']), hp['n_rnn']) # from HO thalamus
                        ],
                'post': [range(0, int(0.75 * hp['n_rnn'])), # to all ctx
                         range(0, int(0.75 * hp['n_rnn'])),  # to all ctx
                         range(0, hp['n_rnn']), # to all ctx + TRN + thalamus
                         range(int(0.85 * hp['n_rnn']), hp['n_rnn']), # to exc thalamus
                         range(0, int(0.85 * hp['n_rnn'])), # to all ctx + TRN
                         range(0, int(0.85 * hp['n_rnn'])) # to all ctx + TRN
                         ],
                'rec': [True, True, True, True, False, False],
                'EI_balance': [True for i in range(6)],
                'exc_prop': [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]},
            'output': {
                'n_modules': 1,
                'pre': ['all'],
                'post': [range(int(hp['n_rnn'] * (1.75 - hp['exc_prop_RNN'])/2), int(0.75 * hp['n_rnn']))], # from CT ctx
                'rec': [False],
                'EI_balance': [False],
                'exc_prop': [None]}
        }
        }

    network_architectures['basic_TC_exc_in_out'] = network_architectures['basic_TC']
    network_architectures['EI_basic_TC_exc_in'] = network_architectures['basic_TC']
    network_architectures['basic_EI_TC_with_TRN'] = network_architectures['basic_TC']
    network_architectures['full_EI_CC_TC_with_TRN_v2'] = network_architectures['basic_TC']

    if 'exc_input_and_output' in hp and hp['exc_input_and_output']:
        for layer in ['sen_input', 'rule_input', 'output']:
            network_architectures[hp['w_mask_type']][layer]['EI_balance'] = [True for i in range(
                network_architectures[hp['w_mask_type']][layer]['n_modules'])]
            network_architectures[hp['w_mask_type']][layer]['exc_prop'] = [1. for i in range(
                network_architectures[hp['w_mask_type']][layer]['n_modules'])]

    if 'exc_inh_RNN' in hp and hp['exc_inh_RNN']:
        network_architectures[hp['w_mask_type']]['rnn']['EI_balance'] = [True for i in range(
            network_architectures[hp['w_mask_type']]['rnn']['n_modules'])]

        if hp['w_mask_type'] == 'EI_basic_TC_exc_in':
            network_architectures[hp['w_mask_type']]['rnn']['exc_prop'] = [hp['exc_prop_RNN'] if i < 4 else 1. for i in range(
                network_architectures[hp['w_mask_type']]['rnn']['n_modules'])]

        elif hp['w_mask_type'] == 'basic_EI_TC_with_TRN':

            n_modules = 10
            pre = []
            post = []
            exc_prop = []

            n_count = 0
            for module in range(n_modules):
                if module < 9:
                    if module % 2 == 0:
                        pre.append(range(n_count, n_count+n_inh_units_module))
                        post.append(range(n_count+n_inh_units_module, n_count+n_inh_units_module+n_exc_units_module))
                        exc_prop.append(0.)
                        n_count += n_inh_units_module
                    else:
                        pre.append(range(n_count, n_count + n_exc_units_module))
                        post.append(np.concatenate([range(n_count-n_inh_units_module, n_count), range(4 * int(hp['n_rnn'] / 5), hp['n_rnn'])]))
                        exc_prop.append(1.)
                        n_count += n_exc_units_module
                elif module == 9:
                    pre.append(range(n_count, n_count + n_exc_units_module))
                    post.append(range(0, n_count))
                    exc_prop.append(1.)
                    n_count += n_exc_units_module

            network_architectures[hp['w_mask_type']]['rnn'] = {
                'n_modules': n_modules,
                'pre': pre,
                'post': post,
                'rec': [True if i < 9 else False for i in range(n_modules)],
                'EI_balance': [True for i in range(n_modules)],
                'exc_prop': exc_prop

            }
            network_architectures[hp['w_mask_type']]['output'] = {
                'n_modules': 1,
                'pre': ['all'],
                'post': [range(3 * int(hp['n_rnn'] / 5) + n_inh_units_module, 4 * int(hp['n_rnn'] / 5))],
                'rec': [False],
                'EI_balance': [True],
                'exc_prop': [1.]
            }


        elif hp['w_mask_type'] == 'full_EI_CC_TC_with_TRN_v2':

            n_modules = 20

            pre = []
            post = []
            exc_prop = []

            n_count = 0
            for module in range(n_modules):
                if module < 12:
                    if module % 3 == 0:
                        pre.append(range(n_count, n_count + n_inh_units_module))
                        post.append(range(n_count+n_inh_units_module, n_count+n_inh_units_module+n_exc_units_module))
                        exc_prop.append(0.)
                        n_count += n_inh_units_module
                    elif module % 3 == 1:
                        pre.append(range(n_count, n_count + int(n_exc_units_module/2)))
                        post_ranges = [range(n_count - n_inh_units_module, n_count+n_exc_units_module)]
                        if module == 1:
                            post_ranges.append(range(int(hp['n_rnn'] / 5), 4 * int(hp['n_rnn'] / 5)))
                        elif module == 4 or module == 7:
                            post_ranges.append(range(0, int(hp['n_rnn'] / 5)))
                        post.append(np.concatenate(post_ranges))
                        exc_prop.append(1.)
                        n_count += int(n_exc_units_module/2)
                    elif module % 3 == 2:
                        pre.append(range(n_count, n_count + int(n_exc_units_module / 2)))
                        post_ranges = [range(n_count - n_inh_units_module - int(n_exc_units_module / 2), n_count + int(n_exc_units_module / 2))]
                        if module == 2:
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5), 4 * int(hp['n_rnn'] / 5) + n_TRN_units))
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5) + n_TRN_units, 4 * int(hp['n_rnn'] / 5) + n_TRN_units + int(n_exc_thal_units / 4)))
                        elif module == 5:
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5) + int(n_TRN_units / 4), 4 * int(hp['n_rnn'] / 5) + 2 * int(n_TRN_units / 4)))
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5) + n_TRN_units + int(n_exc_thal_units / 4), 4 * int(hp['n_rnn'] / 5) + n_TRN_units + 2 * int(n_exc_thal_units / 4)))
                        elif module == 8:
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5) + 2 * int(n_TRN_units / 4), 4 * int(hp['n_rnn'] / 5) + 3 * int(n_TRN_units / 4)))
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5) + n_TRN_units + 2 * int(n_exc_thal_units / 4), 4 * int(hp['n_rnn'] / 5) + n_TRN_units + 3 * int(n_exc_thal_units / 4)))
                        elif module == 11:
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5) + 3 * int(n_TRN_units / 4), 4 * int(hp['n_rnn'] / 5) + n_TRN_units))
                            post_ranges.append(range(4 * int(hp['n_rnn'] / 5) + n_TRN_units + 3 * int(n_exc_thal_units / 4), 4 * int(hp['n_rnn'] / 5) + n_TRN_units + n_exc_thal_units))
                        post.append(np.concatenate(post_ranges))
                        exc_prop.append(1.)
                        n_count += int(n_exc_units_module / 2)

                elif module >= 12 and module < 16:
                    pre.append(range(n_count, n_count + int(n_TRN_units / 4)))
                    post_range_start = 4 * int(hp['n_rnn'] / 5) + n_TRN_units + (module % 12) * int(n_exc_thal_units / 4)
                    post_range_end = post_range_start + int(n_exc_thal_units / 4)
                    post.append(range(post_range_start,post_range_end))
                    exc_prop.append(0.)
                    n_count += int(n_TRN_units / 4)
                elif module >= 16:
                    pre.append(range(n_count, n_count + int(n_exc_thal_units / 4)))
                    post_range_start1 = (module % 16) * int(hp['n_rnn'] / 5)
                    post_range_end1 = post_range_start1 + int(hp['n_rnn'] / 5)
                    post_range_start2 = 4 * int(hp['n_rnn'] / 5) + (module % 16) * int(n_TRN_units / 4)
                    post_range_end2 = post_range_start2 + int(n_TRN_units / 4)
                    post.append(np.concatenate([range(post_range_start1,post_range_end1), range(post_range_start2, post_range_end2)]))
                    exc_prop.append(1.)
                    n_count += int(n_exc_thal_units / 4)

            network_architectures[hp['w_mask_type']]['sen_input']['post'] = [range(0, int(hp['n_rnn'] / 5)),
                                                                             range(4 * int(hp['n_rnn'] / 5) + n_TRN_units + int(n_exc_thal_units / 4),
                                                                                   4 * int(hp['n_rnn'] / 5) + n_TRN_units + 2 * int(n_exc_thal_units / 4)),
                                                                             range(4 * int(hp['n_rnn'] / 5) + n_TRN_units + 2 * int(n_exc_thal_units / 4),
                                                                                   4 * int(hp['n_rnn'] / 5) + n_TRN_units + 3 * int(n_exc_thal_units / 4))
                                                                             ]

            network_architectures[hp['w_mask_type']]['rnn'] = {
                'n_modules': n_modules,
                'pre': pre,
                'post': post,
                'rec': [True if i < 16 else False for i in range(n_modules)],
                'EI_balance': [True for i in range(n_modules)],
                'exc_prop': exc_prop

            }
            network_architectures[hp['w_mask_type']]['output'] = {
                'n_modules': 1,
                'pre': ['all'],
                'post': [range(3 * int(hp['n_rnn'] / 5) + n_inh_units_module + int(n_exc_units_module / 2),
                               4 * int(hp['n_rnn'] / 5))],
                'rec': [False],
                'EI_balance': [True],
                'exc_prop': [1.]
            }


    return network_architectures

def get_network_modules_params(hp):
    """

    Args:
        hp: network hyperparameters

    Returns:
        module_params: a list of dictionaries with the parameters of the various network modules

    """

    module_params = []

    if 'use_w_mask' in hp and hp['use_w_mask'] and 'w_mask_type' in hp and hp['w_mask_type']: # and 'random_connectivity' in hp and not hp['random_connectivity']:

        net_arc = all_network_architectures(hp)[hp['w_mask_type']]

        for layer_type, modules in net_arc.items():
            for m in range(modules['n_modules']):
                module_params.append({
                    'layer': layer_type,
                    'pre_node_indexes': modules['pre'][m],
                    'post_node_indexes': modules['post'][m],
                    'keep_recurrency': modules['rec'][m],
                    'EI_balance': modules['EI_balance'][m],
                    'exc_prop': modules['exc_prop'][m]
                })

    else:
        module_params.append({
            'layer': 'sen_input',
            'pre_node_indexes': range(0, 1+hp['num_ring']*hp['n_eachring']),
            'post_node_indexes': range(0, hp['n_rnn']),
            'keep_recurrency': False,
            'EI_balance': hp['exc_input_and_output'],
            'exc_prop': 1 if hp['exc_input_and_output'] else None
        })
        module_params.append({
            'layer': 'rule_input',
            'pre_node_indexes': range(0, hp['n_rule']),
            'post_node_indexes': range(0, hp['n_rnn']),
            'keep_recurrency': False,
            'EI_balance': hp['exc_input_and_output'],
            'exc_prop': 1 if hp['exc_input_and_output'] else None
        })
        module_params.append({
            'layer': 'rnn',
            'pre_node_indexes': range(0, hp['n_rnn']),
            'post_node_indexes': range(0, hp['n_rnn']),
            'keep_recurrency': True,
            'EI_balance': hp['exc_input_and_output'],
            'exc_prop': hp['exc_prop_RNN'] if hp['exc_input_and_output'] else None
        })
        module_params.append({
            'layer': 'output',
            'pre_node_indexes': range(0, hp['n_output']),
            'post_node_indexes': range(0, hp['n_rnn']),
            'keep_recurrency': False,
            'EI_balance': hp['exc_input_and_output'],
            'exc_prop': 1 if hp['exc_input_and_output'] else None
        })

    return module_params


def generate_weight_masks_and_EI_lists(hp, module_params):
    """Generates a dict with masks of connectivity and excitation/inhibition to be applied later on weight matrices

    Args:
        hp: network hyperparameters

    Returns:
        w_masks: Dictionary with masks for each type of weight matrix
        EI_lists: Dictionary with lists indicating if units are excitatory or inhibitory
    """

    layers_n_units = {'sen_input': 1 + hp['num_ring'] * hp['n_eachring'],
                      'rule_input': hp['n_rule'],
                      'rnn': hp['n_rnn'],
                      'output': hp['n_output']
                      }

    w_mask = {layer: np.ones([n_units, hp['n_rnn']]) for layer, n_units in layers_n_units.items()}
    EI_lists = {layer: np.zeros(n_units) for layer, n_units in layers_n_units.items()}
    EI_lists['input'] = np.zeros(hp['n_input'])

    if 'use_w_mask' in hp and hp['use_w_mask']:

        print('\nNetwork based on ' + hp['w_mask_type'] + ' weight mask.\n')

        EI_lists_rnn = []
        for m in module_params:
            w_mask[m['layer']] = reduce_weight_matrix(w_mask[m['layer']],
                                                      pre_node_indexes=m['pre_node_indexes'],
                                                      post_node_indexes=m['post_node_indexes'],
                                                      keep_recurrency=m['keep_recurrency'])

            if 'exc_inh_RNN' in hp and hp['exc_inh_RNN'] and 'rnn' in m['layer']:
                n_unit_module = len(m['pre_node_indexes'])
                exc_prop_module = m['exc_prop']
                n_inh_units = n_unit_module - int(n_unit_module * exc_prop_module)
                EI_list_module = np.ones(n_unit_module)
                EI_list_module[:n_inh_units] = -1
                EI_lists_rnn.append(EI_list_module)

        EI_lists['rnn'] = np.concatenate(EI_lists_rnn, axis=0)

        if 'random_connectivity' in hp and hp['random_connectivity']:
            print('\nRandomised weight mask used.\n')

            for layer, n_units in layers_n_units.items():
                w_mask[layer] = sparsify_weight_matrix(
                    np.ones([n_units, hp['n_rnn']]),
                    perc_weights_to_zero=sum(w_mask[layer].flatten() == 0).astype('int') / len(w_mask[layer].flatten()),
                    seed=hp['seed'])

    else:
        print('\nNo weight mask used.\n')

        if 'exc_inh_RNN' in hp and hp['exc_inh_RNN']:
            n_inh_units = hp['n_rnn'] - int(hp['n_rnn']*hp['exc_prop_RNN'])
            EI_list_rnn = np.ones(hp['n_rnn'])
            EI_list_rnn[:n_inh_units] = -1
            EI_lists['rnn'] = EI_list_rnn

    w_mask['input'] = np.concatenate([w_mask['sen_input'], w_mask['rule_input']], axis=0)
    w_mask['output'] = w_mask['output'].T

    if 'exc_input_and_output' in hp and hp['exc_input_and_output']:
        for layer, n_units in layers_n_units.items():
            if layer is not 'rnn':
                EI_lists[layer] = np.ones(n_units)
        EI_lists['input'] = np.ones(hp['n_input'])

    if 'exc_inh_RNN' in hp and hp['exc_inh_RNN']:
        EI_lists['output'] = EI_lists['rnn']

    return w_mask, EI_lists


def reduce_weight_matrix(weights, pre_node_indexes='all', post_node_indexes='all', keep_recurrency=False):
    if pre_node_indexes == 'all':
        pre_node_indexes = range(weights.shape[0])
    if post_node_indexes == 'all':
        post_node_indexes = range(weights.shape[1])

    weights_to_zero = np.setdiff1d(range(weights[0].shape[0]), post_node_indexes)

    if keep_recurrency:
        if weights.shape[0] == weights.shape[1]:
            weights_to_zero = np.setdiff1d(weights_to_zero, pre_node_indexes)
        else:
            print('Weight matrix is not squared!')

    if len(weights_to_zero.shape):
        weights[np.ix_(pre_node_indexes, weights_to_zero)] = 0

    return weights


def sparsify_weight_matrix(weights, perc_weights_to_zero, seed):
    np.random.seed(seed)

    n_weights_to_zero = int(weights.shape[1] * perc_weights_to_zero)

    all_indexes = [[(i, j) for j in range(weights.shape[1])] for i in range(weights.shape[0])]

    for row_indexes in all_indexes:
        sampled_indexes = np.array([row_indexes[i] for i in np.random.choice(range(len(row_indexes)),
                                                                             size=n_weights_to_zero,
                                                                             replace=False)]).T
        weights[tuple(sampled_indexes)] = 0

    return weights


class LeakyRNNCell(RNNCell):
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 n_input,
                 alpha,
                 w_mask_input,
                 w_mask_rnn,
                 EI_list_input=None,
                 EI_list_rnn=None,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._n_input = n_input
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrices
        n_hidden = self._num_units

        # Gaussian with mean 0.0
        w_in0 = (self._w_in_start *
                 self.rng.randn(n_input, n_hidden) / np.sqrt(n_input))

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                              rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))

        # Apply weight mask matrix to input and rnn weights
        w_in0 = w_in0 * w_mask_input
        w_rec0 = w_rec0 * w_mask_rnn

        # Apply EI masks if used
        if EI_list_input is None:
            self.EI_input = False
        else:
            w_in0 = np.matmul(np.diag(EI_list_input), np.abs(w_in0))
            self.EI_input = True
            self._EI_matrix_input = tf.constant(np.diag(EI_list_input).astype('float32'))

        if EI_list_rnn is None:
            self.EI_rnn = False
        else:
            w_rec0 = np.matmul(np.diag(EI_list_rnn), np.abs(w_rec0))
            self.EI_rnn = True
            self._EI_matrix_rnn = tf.constant(np.diag(EI_list_rnn).astype('float32'))

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.w_rnn0 = matrix0
        self._initializer = tf.compat.v1.constant_initializer(matrix0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1]
        self._kernel = self.add_variable(
                'kernel',
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._initializer)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        if self.EI_input or self.EI_rnn:
            w_in, w_rec = tf.split(self._kernel, [self._n_input, self._num_units], axis=0)

            if self.EI_input:
                w_in = tf.matmul(self._EI_matrix_input, tf.abs(w_in))

            if self.EI_rnn:
                w_rec = tf.matmul(self._EI_matrix_rnn, tf.abs(w_rec))

            self._kernel = tf.concat([w_in, w_rec],0)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random.normal(tf.shape(input=state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        return output, output


class LeakyGRUCell(RNNCell):
  """Leaky Gated Recurrent Unit cell (cf. https://elifesciences.org/articles/21492).

  Args:
    num_units: int, The number of units in the GRU cell.
    alpha: dt/T, simulation time step over time constant
    sigma_rec: recurrent noise
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               alpha,
               sigma_rec=0,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(LeakyGRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    # self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

    self._alpha = alpha
    self._sigma = np.sqrt(2 / alpha) * sigma_rec

    # TODO(gryang): allow this to use different initialization

  @property
  def state_size(self):
      return self._num_units

  @property
  def output_size(self):
      return self._num_units

  def build(self, inputs_shape):
      if inputs_shape[1] is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                         % inputs_shape)

      input_depth = inputs_shape[1]
      self._gate_kernel = self.add_variable(
          "gates/%s" % 'kernel',
          shape=[input_depth + self._num_units, 2 * self._num_units],
          initializer=self._kernel_initializer)
      self._gate_bias = self.add_variable(
          "gates/%s" % 'bias',
          shape=[2 * self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.constant_initializer(1.0, dtype=self.dtype)))
      self._candidate_kernel = self.add_variable(
          "candidate/%s" % 'kernel',
          shape=[input_depth + self._num_units, self._num_units],
          initializer=self._kernel_initializer)
      self._candidate_bias = self.add_variable(
          "candidate/%s" % 'bias',
          shape=[self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.zeros_initializer(dtype=self.dtype)))

      self.built = True

  def call(self, inputs, state):
      """Gated recurrent unit (GRU) with nunits cells."""

      gate_inputs = math_ops.matmul(
          array_ops.concat([inputs, state], 1), self._gate_kernel)
      gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

      value = math_ops.sigmoid(gate_inputs)
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

      r_state = r * state

      candidate = math_ops.matmul(
          array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
      candidate = nn_ops.bias_add(candidate, self._candidate_bias)
      candidate += tf.random.normal(tf.shape(input=state), mean=0, stddev=self._sigma)

      c = self._activation(candidate)
      # new_h = u * state + (1 - u) * c  # original GRU
      new_h = (1 - self._alpha * u) * state + (self._alpha * u) * c

      return new_h, new_h


class LeakyRNNCellSeparateInput(RNNCell):
    """The most basic RNN cell with external inputs separated.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 alpha,
                 w_mask_rnn,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCellSeparateInput, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                              rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))
        else:
            raise ValueError

        # Apply weight matrix
        w_rec0 = w_rec0 * w_mask_rnn

        self.w_rnn0 = w_rec0
        self._initializer = tf.compat.v1.constant_initializer(w_rec0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        self._kernel = self.add_variable(
                'kernel',
                shape=[self._num_units, self._num_units],
                initializer=self._initializer)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """output = new_state = act(input + U * state + B)."""

        gate_inputs = math_ops.matmul(state, self._kernel)
        gate_inputs = gate_inputs + inputs  # directly add inputs
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random.normal(tf.shape(input=state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        return output, output



class Model(object):
    """The model."""

    def __init__(self,
                 model_dir,
                 hp=None,
                 sigma_rec=None,
                 dt=None):
        """
        Initializing the model with information from hp

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """

        # Reset tensorflow graphs
        tf.compat.v1.reset_default_graph()  # must be in the beginning

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.compat.v1.set_random_seed(hp['seed'])
        self.rng = np.random.RandomState(hp['seed'])

        if sigma_rec is not None:
            print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hp['sigma_rec'] = sigma_rec

        if dt is not None:
            print('Overwrite original dt with {:0.1f}'.format(dt))
            hp['dt'] = dt

        hp['alpha'] = 1.0*hp['dt']/hp['tau']

        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        if hp['in_type'] != 'normal':
            raise ValueError('Only support in_type ' + hp['in_type'])

        # 'random' is no longer a mask type but a control randomisation matched to a given architecture
        if hp['w_mask_type'] == 'random':
            hp['w_mask_type'] = 'basic_TC'
            hp['random_connectivity'] = False

        self.hp = hp
        self._build(hp)

        self.model_dir = model_dir
        self.hp = hp


    def _build(self, hp):

        # Get list of module parameters (for modular RNNs)
        self.module_params = get_network_modules_params(hp)

        # Generate dict with weight masks
        # (matrices of ones if hp['use_w_mask'] is False)
        # and dict of arrays specifying which units are exc (1.) or inh (-1.)
        # (array of zeros if hp['exc_input_and_output'] and/or hp['exc_inh_RNN'] is False)
        self.w_masks_all, self.EI_lists = generate_weight_masks_and_EI_lists(hp, self.module_params)

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.compat.v1.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(input_tensor=tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(input_tensor=tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.compat.v1.train.AdamOptimizer(
                learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=hp['learning_rate'])
        # Set cost
        self.set_optimizer()

        # Variable saver
        # self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.compat.v1.train.Saver()

    def _build_fused(self, hp):
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = tf.compat.v1.placeholder("float", [None, None, n_input])
        self.y = tf.compat.v1.placeholder("float", [None, None, n_output])
        if hp['loss_type'] == 'lsq':
            self.c_mask = tf.compat.v1.placeholder("float", [None, n_output])
        else:
            # Mask on time
            self.c_mask = tf.compat.v1.placeholder("float", [None])

        # Activation functions
        if hp['activation'] == 'power':
            f_act = lambda x: tf.square(tf.nn.relu(x))
        elif hp['activation'] == 'retanh':
            f_act = lambda x: tf.tanh(tf.nn.relu(x))
        elif hp['activation'] == 'relu+':
            f_act = lambda x: tf.nn.relu(x + tf.constant(1.))
        else:
            f_act = getattr(tf.nn, hp['activation'])

        # EI matrices
        if 'exc_input_and_output' in hp and hp['exc_input_and_output']:
            EI_list_input = self.EI_lists['input']
        else:
            EI_list_input = None

        if 'exc_inh_RNN' in hp and hp['exc_inh_RNN']:
            EI_list_rnn = self.EI_lists['rnn']
        else:
            EI_list_rnn = None

        # Recurrent activity
        if hp['rnn_type'] == 'LeakyRNN':
            n_in_rnn = self.x.get_shape().as_list()[-1]
            cell = LeakyRNNCell(n_rnn, n_in_rnn,
                                hp['alpha'],
                                w_mask_input=self.w_masks_all['input'],
                                w_mask_rnn=self.w_masks_all['rnn'],
                                EI_list_input=EI_list_input,
                                EI_list_rnn=EI_list_rnn,
                                sigma_rec=hp['sigma_rec'],
                                activation=hp['activation'],
                                w_rec_init=hp['w_rec_init'],
                                rng=self.rng)
        elif hp['rnn_type'] == 'LeakyGRU':
            cell = LeakyGRUCell(
                n_rnn, hp['alpha'],
                sigma_rec=hp['sigma_rec'], activation=f_act)
        elif hp['rnn_type'] == 'LSTM':
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_rnn, activation=f_act)

        elif hp['rnn_type'] == 'GRU':
            cell = tf.compat.v1.nn.rnn_cell.GRUCell(n_rnn, activation=f_act)
        else:
            raise NotImplementedError("""rnn_type must be one of LeakyRNN,
                    LeakyGRU, EILeakyGRU, LSTM, GRU
                    """)

        if 'transfer_h_across_trials' in hp and hp['transfer_h_across_trials']:
            self.h_init = tf.compat.v1.placeholder("float", [None, n_rnn])
        else:
            self.h_init = None

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            cell, self.x, dtype=tf.float32, time_major=True, initial_state=self.h_init)

        # Gaussian with mean 0.0 and with output weight mask applied
        w_out0 = (self.rng.randn(n_rnn, n_output) / np.sqrt(n_output)) * self.w_masks_all['output']

        # Apply EI masks if used
        if 'exc_inh_RNN' in hp and hp['exc_inh_RNN']:
            w_out0 = np.matmul(np.diag(self.EI_lists['output']), np.abs(w_out0))
        else:
            if 'exc_input_and_output' in hp and hp['exc_input_and_output']:
                w_out0 = np.matmul(np.abs(w_out0), np.diag(self.EI_lists['output']))

        out_w_initializer = tf.compat.v1.constant_initializer(w_out0, dtype=tf.float32)

        # Output
        with tf.compat.v1.variable_scope("output"):
            # NOT using default initialization `glorot_uniform_initializer` (SVL)
            w_out = tf.compat.v1.get_variable(
                'weights',
                [n_rnn, n_output],
                dtype=tf.float32,
                initializer=out_w_initializer
            )
            b_out = tf.compat.v1.get_variable(
                'biases',
                [n_output],
                dtype=tf.float32,
                initializer=tf.compat.v1.constant_initializer(0.0, dtype=tf.float32)
            )

        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))

        if 'exc_inh_RNN' in hp and hp['exc_inh_RNN']:
            w_out = tf.matmul(tf.constant(np.diag(self.EI_lists['output']).astype('float32')), tf.abs(w_out))
        else:
            if 'exc_input_and_output' in hp and hp['exc_input_and_output']:
                w_out = tf.matmul(tf.abs(w_out), tf.constant(np.diag(self.EI_lists['output']).astype('float32')))

        # y_hat_ shape (n_time*n_batch, n_unit)
        y_hat_ = tf.matmul(h_shaped, w_out) + b_out
        if hp['loss_type'] == 'lsq':
            # Least-square loss
            y_hat = tf.sigmoid(y_hat_)
            self.cost_lsq = tf.reduce_mean(
                input_tensor=tf.square((y_shaped - y_hat) * self.c_mask))
        else:
            y_hat = tf.nn.softmax(y_hat_)
            # Cross-entropy loss
            self.cost_lsq = tf.reduce_mean(
                input_tensor=self.c_mask * tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(y_shaped), logits=y_hat_))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(input=self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_fused(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']
        self.w_masks_labelled = {}

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    # TODO(gryang): For GRU, fix
                    self.w_rec = v[n_input:, :]
                    self.w_in = v[:n_input, :]
                    self.w_masks_labelled[v.name] = np.concatenate((self.w_masks_all['input'],
                                                                    self.w_masks_all['rnn']),
                                                                   axis=0)
                else:
                    self.b_rec = v
                    self.w_masks_labelled[v.name] = np.ones(v.shape)
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                    self.w_masks_labelled[v.name] = self.w_masks_all['output']
                else:
                    self.b_out = v
                    self.w_masks_labelled[v.name] = np.ones(v.shape)

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_in.shape != (n_input, n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((n_input, n_rnn)) + ', but found ' +
                             str(self.w_in.shape))

    def _build_seperate(self, hp):
        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        n_input = hp['n_input']
        n_sen_input = hp['rule_start']
        n_rule_input = hp['n_rule']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = tf.compat.v1.placeholder("float", [None, None, n_input])
        self.y = tf.compat.v1.placeholder("float", [None, None, n_output])
        self.c_mask = tf.compat.v1.placeholder("float", [None, n_output])

        # Gaussian with mean 0.0 and with input weight mask applied
        w_sen_in0 = (self.rng.randn(n_sen_input, n_rnn) / np.sqrt(n_sen_input)) * self.w_masks_all['sen_input']
        sen_w_initializer = tf.compat.v1.constant_initializer(w_sen_in0, dtype=tf.float32)

        sensory_inputs, rule_inputs = tf.split(
            self.x, [hp['rule_start'], hp['n_rule']], axis=-1)

        sensory_rnn_inputs = tf.compat.v1.layers.dense(sensory_inputs, n_rnn, name='sen_input',
                                                       kernel_initializer=sen_w_initializer)

        if 'mix_rule' in hp and hp['mix_rule'] is True:
            # rotate rule matrix
            kernel_initializer = tf.compat.v1.orthogonal_initializer()
            rule_inputs = tf.compat.v1.layers.dense(
                rule_inputs, hp['n_rule'], name='mix_rule',
                use_bias=False, trainable=False,
                kernel_initializer=kernel_initializer)

        # Gaussian with mean 0.0 and with input weight mask applied
        w_rule_in0 = (self.rng.randn(n_rule_input, n_rnn) / np.sqrt(n_rule_input)) * self.w_masks_all['rule_input']
        rule_w_initializer = tf.compat.v1.constant_initializer(w_rule_in0, dtype=tf.float32)

        rule_rnn_inputs = tf.compat.v1.layers.dense(rule_inputs, n_rnn, name='rule_input', use_bias=False,
                                                    kernel_initializer=rule_w_initializer)

        rnn_inputs = sensory_rnn_inputs + rule_rnn_inputs

        # Recurrent activity
        cell = LeakyRNNCellSeparateInput(
            n_rnn, hp['alpha'],
            w_mask_rnn=self.w_masks_all['rnn'],
            sigma_rec=hp['sigma_rec'],
            activation=hp['activation'],
            w_rec_init=hp['w_rec_init'],
            rng=self.rng)

        if 'transfer_h_across_trials' in hp and hp['transfer_h_across_trials']:
            self.h_init = tf.compat.v1.placeholder("float", [None, n_rnn])
        else:
            self.h_init = None

            # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            cell, rnn_inputs, dtype=tf.float32, time_major=True, initial_state=self.h_init)

        # Output
        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))

        # Gaussian with mean 0.0
        w_out0 = (self.rng.randn(n_rnn, n_output) / np.sqrt(n_output)) * self.w_masks_all['output']
        out_w_initializer = tf.compat.v1.constant_initializer(w_out0, dtype=tf.float32)

        # y_hat shape (n_time*n_batch, n_unit)
        y_hat = tf.compat.v1.layers.dense(
            h_shaped, n_output, activation=tf.nn.sigmoid, name='output', kernel_initializer=out_w_initializer)
        # Least-square loss
        self.cost_lsq = tf.reduce_mean(
            input_tensor=tf.square((y_shaped - y_hat) * self.c_mask))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(input=self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_separate(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']
        self.w_masks_labelled = {}

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_rec = v
                    self.w_masks_labelled[v.name] = self.w_masks_all['rnn']
                else:
                    self.b_rec = v
                    self.w_masks_labelled[v.name] = np.ones(v.shape)
            elif 'sen_input' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_sen_in = v
                    self.w_masks_labelled[v.name] = self.w_masks_all['sen_input']
                else:
                    self.b_in = v
                    self.w_masks_labelled[v.name] = np.ones(v.shape)
            elif 'rule_input' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_rule = v
                    self.w_masks_labelled[v.name] = self.w_masks_all['rule_input']
                else:
                    self.b_in_rule = v
                    self.w_masks_labelled[v.name] = np.ones(v.shape)
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                    self.w_masks_labelled[v.name] = self.w_masks_all['output']
                else:
                    self.b_out = v
                    self.w_masks_labelled[v.name] = np.ones(v.shape)

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_sen_in.shape != (hp['rule_start'], n_rnn):
            raise ValueError('Shape of w_sen_in should be ' +
                             str((hp['rule_start'], n_rnn)) + ', but found ' +
                             str(self.w_sen_in.shape))
        if self.w_rule.shape != (hp['n_rule'], n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((hp['n_rule'], n_rnn)) + ', but found ' +
                             str(self.w_rule.shape))

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.compat.v1.get_default_session()
        sess.run(tf.compat.v1.global_variables_initializer())

    def restore(self, load_dir=None):
        """restore the model"""
        sess = tf.compat.v1.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.compat.v1.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
        """Save the model."""
        sess = tf.compat.v1.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def set_optimizer(self, extra_cost=None, var_list=None):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : tensorflow variable,
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = self.var_list

        print('Variables being optimized:')
        for v in var_list:
            print(v)

        self.grads_and_vars = self.opt.compute_gradients(cost, var_list)

        # gradient clipping and applying of weight masks
        if 'use_w_mask' in self.hp and self.hp['use_w_mask']:
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.) * tf.constant(self.w_masks_labelled[var.name].astype('float32')), var)
                          for grad, var in self.grads_and_vars]
        else:
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                          for grad, var in self.grads_and_vars]

        self.train_step = self.opt.apply_gradients(capped_gvs)

    def set_EI_masks(self, sess):
        """Apply EI masks for proper saving of weight variables
        """

        hp = self.hp
        n_input = hp['n_input']
        EI_lists = self.EI_lists

        for v in self.var_list:
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                v_val = sess.run(v)
                if 'rnn' in v.name:
                    if hp['exc_input_and_output']:
                        v_val[:n_input] = np.matmul(np.diag(EI_lists['input']), np.abs(v_val[:n_input]))

                    if hp['exc_inh_RNN']:
                        v_val[n_input:] = np.matmul(np.diag(EI_lists['rnn']), np.abs(v_val[n_input:]))

                elif 'output' in v.name:
                    if hp['exc_inh_RNN']:
                        v_val = np.matmul(np.diag(EI_lists['output']), np.abs(v_val))
                    else:
                        if hp['exc_input_and_output']:
                            v_val = np.matmul(np.abs(v_val), np.diag(EI_lists['output']))
                sess.run(v.assign(v_val))

    def lesion_units(self, sess, units, verbose=False):
        """Lesion units given by units

        Args:
            sess: tensorflow session
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hp['n_input']
        for v in self.var_list:
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                v_val = sess.run(v)
                if 'output' in v.name:
                    # output weights
                    v_val[units, :] = 0
                elif 'rnn' in v.name:
                    # recurrent weights
                    v_val[n_input + units, :] = 0
                sess.run(v.assign(v_val))

        if verbose:
            print('Lesioned units:')
            print(units)

    def lesion_weights(self, sess, pre_units, post_units):
        """Lesion units given by units

        Args:
            sess: tensorflow session
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if pre_units is None:
            return
        elif not hasattr(pre_units, '__iter__'):
            pre_units = np.array([pre_units])
        else:
            pre_units = np.array(pre_units)

        if post_units is None:
            return
        elif not hasattr(post_units, '__iter__'):
            post_units = np.array([post_units])
        else:
            post_units = np.array(post_units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hp['n_input']
        for v in self.var_list:
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                v_val = sess.run(v)
                if 'rnn' in v.name:
                    # recurrent weights
                    for i in pre_units:
                        v_val[n_input + i, post_units] = 0
                sess.run(v.assign(v_val))

