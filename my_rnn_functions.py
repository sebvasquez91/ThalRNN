from os.path import join, basename, dirname, abspath
from os import walk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import train, task
from network import Model, get_perf
from scipy.spatial.distance import cdist
plt.rcParams.update({'font.size': 20})
from scipy.stats import ttest_ind
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import euclidean_distances
import copy

import tools
from analysis import performance
from analysis import standard_analysis
from analysis import clustering
from analysis import variance
from analysis import taskset
from analysis import varyhp
from analysis import data_analysis
from analysis import contextdm_analysis
from analysis import posttrain_analysis

rule_name    = {'reactgo': 'RT Go',
                'delaygo': 'Dly Go',
                'fdgo': 'Go',
                'dm1': 'DM 1',
                'dm2': 'DM 2',
                'contextdm1': 'Ctx DM 1',
                'contextdm2': 'Ctx DM 2',
                'multidm': 'MultSen DM',
                'delaydm1': 'Dly DM 1',
                'delaydm2': 'Dly DM 2',
                'contextdelaydm1': 'Ctx Dly DM 1',
                'contextdelaydm2': 'Ctx Dly DM 2',
                'multidelaydm': 'MultSen Dly DM',
                'contextdelaydm_MD_task_mod1': 'Ctx Dly DM 1 (MD task)',
                'contextdelaydm_MD_task_mod2': 'Ctx Dly DM 2 (MD task)',
                'reactanti': 'RT Anti',
                'delayanti': 'Dly Anti',
                'fdanti': 'Anti',
                'dmsgo': 'DMS',
                'dmsnogo': 'DNMS',
                'dmcgo': 'DMC',
                'dmcnogo': 'DNMC',
                'oic': '1IC',
                'dmc': 'DMC'
                }


def get_model_data(model_dir = None, get_clusters=False):

    hp = get_hp(model_dir)

    if model_dir is None:
        model_data = {'hp':hp}
        return model_data

    w_rec, w_in, w_out, b_rec, b_out, w_masks_all = get_learnt_weights(model_dir,hp)
    w_sen = w_in[:hp['rule_start'],:]
    w_rules = w_in[hp['rule_start']:,:]

    model_data = {'hp': hp, 'w_rec': w_rec, 'w_in': w_in, 'w_out': w_out,
                  'w_sen': w_sen, 'w_rules': w_rules, 'b_rec': b_rec, 'b_out': b_out,
                  'w_masks_all': w_masks_all}

    return model_data


def get_hp(model_dir=None):
    def_ruleset = 'all' #'ctx_multi_sensory_delay' #'ctx_multi_sensory'
    if model_dir:
        model = Model(model_dir)
        with tf.compat.v1.Session() as sess:
            model.restore()
            hp = model.hp
        print('Model hp restored.')
    else:
        hp = train.get_default_hp(def_ruleset)
        print('Model hp not found. Default hp used.')

    return hp


def subnetwork_indx(subset, n_rnn=100, n_modules=5, prop_inh=0.2, prop_TRN=0.4):
    n_unit_mod = int(n_rnn / n_modules)
    prop_exc_thal = 1 - prop_TRN

    if subset == 'all':
        indx1, idx2 = 0, n_rnn
    elif subset == 'PFC':
        indx1, idx2 = 0 * n_unit_mod, 1 * n_unit_mod
    elif subset == 'PFC_inh':
        indx1, idx2 = 0 * n_unit_mod, 0 * n_unit_mod + int(n_unit_mod*prop_inh)
    elif subset == 'PFC_exc':
        indx1, idx2 = 0 * n_unit_mod + int(n_unit_mod*prop_inh), 1 * n_unit_mod

    elif subset == 'Mod1':
        indx1, idx2 = 1 * n_unit_mod, 2 * n_unit_mod
    elif subset == 'Mod1_inh':
        indx1, idx2 = 1 * n_unit_mod, 1 * n_unit_mod + int(n_unit_mod*prop_inh)
    elif subset == 'Mod1_exc':
        indx1, idx2 = 1 * n_unit_mod + int(n_unit_mod*prop_inh), 2 * n_unit_mod

    elif subset == 'Mod2':
        indx1, idx2 = 2 * n_unit_mod, 3 * n_unit_mod
    elif subset == 'Mod2_inh':
        indx1, idx2 = 2 * n_unit_mod, 2 * n_unit_mod + int(n_unit_mod*prop_inh)
    elif subset == 'Mod2_exc':
        indx1, idx2 = 2 * n_unit_mod + int(n_unit_mod*prop_inh), 3 * n_unit_mod

    elif subset == 'Mot':
        indx1, idx2 = 3 * n_unit_mod, 4 * n_unit_mod
    elif subset == 'Mot_inh':
        indx1, idx2 = 3 * n_unit_mod, 3 * n_unit_mod + int(n_unit_mod*prop_inh)
    elif subset == 'Mot_exc':
        indx1, idx2 = 3 * n_unit_mod + int(n_unit_mod*prop_inh), 4 * n_unit_mod

    elif subset == 'TRN':
        indx1, idx2 = 4 * n_unit_mod, 4 * n_unit_mod + int(prop_TRN * n_unit_mod)
    elif subset == 'TRN_MD':
        indx1, idx2 = 4 * n_unit_mod + 0 * int(prop_TRN * n_unit_mod / 4), \
                      4 * n_unit_mod + 1 * int(prop_TRN * n_unit_mod / 4)
    elif subset == 'TRN_Mod1':
        indx1, idx2 = 4 * n_unit_mod + 1 * int(prop_TRN * n_unit_mod / 4), \
                      4 * n_unit_mod + 2 * int(prop_TRN * n_unit_mod / 4)
    elif subset == 'TRN_Mod2':
        indx1, idx2 = 4 * n_unit_mod + 2 * int(prop_TRN * n_unit_mod / 4), \
                      4 * n_unit_mod + 3 * int(prop_TRN * n_unit_mod / 4)
    elif subset == 'TRN_Mot':
        indx1, idx2 = 4 * n_unit_mod + 3 * int(prop_TRN * n_unit_mod / 4), \
                      4 * n_unit_mod + 4 * int(prop_TRN * n_unit_mod / 4)

    elif subset == 'Thal':
        indx1, idx2 = 4 * n_unit_mod + int(prop_TRN * n_unit_mod), 5 * n_unit_mod
    elif subset == 'Thal_MD':
        indx1, idx2 = 4 * n_unit_mod + int(prop_TRN * n_unit_mod) + 0 * int(prop_exc_thal * n_unit_mod / 4), \
                      4 * n_unit_mod + int(prop_TRN * n_unit_mod) + 1 * int(prop_exc_thal * n_unit_mod / 4)
    elif subset == 'Thal_Mod1':
        indx1, idx2 = 4 * n_unit_mod + int(prop_TRN * n_unit_mod) + 1 * int(prop_exc_thal * n_unit_mod / 4), \
                      4 * n_unit_mod + int(prop_TRN * n_unit_mod) + 2 * int(prop_exc_thal * n_unit_mod / 4)
    elif subset == 'Thal_Mod2':
        indx1, idx2 = 4 * n_unit_mod + int(prop_TRN * n_unit_mod) + 2 * int(prop_exc_thal * n_unit_mod / 4), \
                      4 * n_unit_mod + int(prop_TRN * n_unit_mod) + 3 * int(prop_exc_thal * n_unit_mod / 4)
    elif subset == 'Thal_Mot':
        indx1, idx2 = 4 * n_unit_mod + int(prop_TRN * n_unit_mod) + 3 * int(prop_exc_thal * n_unit_mod / 4), \
                      5 * n_unit_mod

    elif subset == 'Thal_sen':
        part1 = subnetwork_indx('Thal_Mod1', n_rnn, n_modules, prop_inh, prop_TRN)
        part2 = subnetwork_indx('Thal_Mod2', n_rnn, n_modules, prop_inh, prop_TRN)
        return list(part1) + list(part2)

    elif subset == 'Thal_non-sen':
        part1 = subnetwork_indx('Thal_MD', n_rnn, n_modules, prop_inh, prop_TRN)
        part2 = subnetwork_indx('Thal_Mot', n_rnn, n_modules, prop_inh, prop_TRN)
        return list(part1) + list(part2)

    return range(indx1, idx2)


def get_subnetwork_dict(n_rnn=100, n_modules=5, prop_inh=0.2, prop_TRN=0.4, subnetworks_to_plot='all'):

    subnetworks = [
        {'name': 'PFC',
         'index_range': subnetwork_indx('PFC', n_rnn, n_modules, prop_inh, prop_TRN)
         # 'pc_projections':
         },
        {'name': 'Thalamus',
         'index_range': subnetwork_indx('Thal', n_rnn, n_modules, prop_inh, prop_TRN)
         },
        {'name': 'TRN',
         'index_range': subnetwork_indx('TRN', n_rnn, n_modules, prop_inh, prop_TRN)
         },
        {'name': 'Mod 1',
         'index_range': subnetwork_indx('Mod1', n_rnn, n_modules, prop_inh, prop_TRN)
         },
        {'name': 'Mod 2',
         'index_range': subnetwork_indx('Mod2', n_rnn, n_modules, prop_inh, prop_TRN)
         },
        {'name': 'Motor',
         'index_range': subnetwork_indx('Mot', n_rnn, n_modules, prop_inh, prop_TRN)
         }
    ]

    if type(subnetworks_to_plot) == str:
        if subnetworks_to_plot != 'all':
            subnetworks = [sn for sn in subnetworks if sn['name'] == subnetworks_to_plot]
    elif type(subnetworks_to_plot) == list:
        subnetworks = [sn for sn in subnetworks if sn['name'] in subnetworks_to_plot]

    return subnetworks

def get_learnt_weights(model_dir, hp):
    n_input = hp['n_input']
    if model_dir:
        model = Model(model_dir)
        with tf.compat.v1.Session() as sess:
            model.restore()
            # get dict with weight masks used
            w_masks_all = model.w_masks_all
            # get all connection weights and biases as tensorflow variables
            var_list = model.var_list
            # evaluate the parameters after training
            trained_vars = [sess.run(var) for var in var_list]
            # get name of each variable
            names  = [var.name for var in var_list]

        if hp['use_separate_input']:
            for v, name in zip(trained_vars, names):
                print(name)
                if 'sen_input' in name and 'kernel' in name:
                    w_sen_in = v
                elif 'rule_input' in name and 'kernel' in name:
                    w_rule_in = v
                elif 'rnn' in name:
                    if 'kernel' in name or 'weight' in name:
                        w_rec = v
                    else:
                        b_rec = v
                elif 'output' in name:
                    if 'kernel' in name or 'weight' in name:
                        w_out = v
                    else:
                        b_out = v
                else:
                    continue

            w_in = np.concatenate([w_sen_in,w_rule_in], axis=0)

            return w_rec, w_in, w_out, b_rec, b_out, w_masks_all

        else:
            for v, name in zip(trained_vars, names):
                if 'rnn' in name:
                    if 'kernel' in name or 'weight' in name:
                        w_rec = v[n_input:, :]
                        w_in = v[:n_input, :]
                    else:
                        b_rec = v
                elif 'output' in name:
                    if 'kernel' in name or 'weight' in name:
                        w_out = v
                    else:
                        b_out = v
                else:
                    continue

            return w_rec, w_in, w_out, b_rec, b_out, w_masks_all


def plot_performanceprogress(model_dir, rule_color, ax=None, fig=None, rule_plot=None, label=None, show_legend=False,
                             average_rules=True, plot_type='perf'):
    # Plot Training Progress
    model_parts = sorted([folder[0] for folder in walk(dirname(abspath(model_dir))) if basename(folder[0]).startswith(basename(model_dir))])
    hp = tools.load_hp(model_parts[0])

    log = {}
    trials = []
    trial_count = 0

    for rule in rule_plot:
        log = {**log, 'cost_'+rule: [], 'perf_'+rule: []}
    for model_part in model_parts:
        temp_log = tools.load_log(model_part)
        trials = trials + [t + trial_count for t in temp_log['trials']]
        trial_count = trials[-1]
        for rule in rule_plot:
            log['perf_' + rule] = log['perf_' + rule] + temp_log['perf_' + rule]
            log['cost_' + rule] = log['cost_' + rule] + temp_log['cost_' + rule]

    if average_rules:
        log = {'perf_rule_average': [sum([log['perf_' + rule][perf_i] for rule in rule_plot])/len(rule_plot) for perf_i in range(len(trials))],
               'cost_rule_average': [sum([log['cost_' + rule][perf_i] for rule in rule_plot])/len(rule_plot) for perf_i in range(len(trials))]}
        rule_plot = ['rule_average']

    fs = 14 # fontsize
    
    if not fig:
        fig = plt.figure(figsize=(15,6))
        
    if not ax:
        ax = fig.add_axes([0.1,0.25,0.35,0.6])
        
    lines = list()
    labels = list()

    x_plot = np.array(trials)/1000.
    if rule_plot == None:
        rule_plot = hp['rules']

    for i, rule in enumerate(rule_plot):
        # line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=color_rules[i%26])
        # ax2.plot(x_plot, perf_tests[rule],color=color_rules[i%26])
        if plot_type == 'perf':
            line = ax.plot(x_plot, log['perf_' + rule], color=rule_color[rule])
        else:
            line = ax.plot(x_plot, np.log10(log['cost_'+rule]), color=rule_color[rule])
        lines.append(line[0])
        if label:
            labels.append(label)
        else:
            if average_rules:
                labels.append('Rule averages')
            else:
                labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)

    ax.set_ylim([0, 1])
    ax.set_xlabel('Total trials (x$10^3$)',fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance',fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=3)
    ax.set_yticks([0,1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if show_legend:
        lg = fig.legend(lines, labels, title='$\lambda$ = 0',ncol=2,bbox_to_anchor=(0.15,0.22),fontsize=fs,labelspacing=0.3,loc=6,frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
    #plt.savefig('figure/Performance_Progresss.pdf', transparent=True)
    return fig, ax
    


def plt_various_performances(trained_models,models_saving_dir='./saved_models',
                             rules=['contextdelaydm_MD_task_mod1', 'contextdelaydm_MD_task_mod2'],
                             average_rules=True, colors_to_use = 'tableau', show_legend=False, labels=None):


    if colors_to_use == 'tableau':
        colors = mcolors.TABLEAU_COLORS
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        rule_color_pairs = [name for hsv, name in by_hsv]
    else:
        rule_color_pairs = ['bright purple','green blue','indigo','grey blue','lavender','aqua']

    if len(trained_models) > len(rule_color_pairs):
        for i in range(int(np.ceil(len(trained_models)/len(rule_color_pairs)))):
            rule_color_pairs = rule_color_pairs + rule_color_pairs

    fig_all = plt.figure(figsize=(15,6))

    custom_lines = []

    for i, trained_model in enumerate(trained_models):
        if average_rules:
            _rule_color = dict.fromkeys(['rule_average'])
        else:
            _rule_color = dict.fromkeys(rules)

        for rule in _rule_color.keys():
            _rule_color[rule] = rule_color_pairs.pop()

        if colors_to_use  == 'tableau':
            rule_color = {k: v for k, v in _rule_color.items()}
        else:
            rule_color = {k: 'xkcd:'+v for k, v in _rule_color.items()}

        custom_lines.append(Line2D([0], [0], color=colors[rule_color[rule]], lw=2))
    
        if i == 0:
            fig = plt.figure(figsize=(15,6))
            ax = None
        else:
            fig = None

        if labels is None:
            label = ''
        else:
            label = labels[i]

        model_dir = join(models_saving_dir,trained_model)
        print(model_dir)
        #try:
        _, ax = plot_performanceprogress(model_dir, fig=fig_all , ax=ax, rule_color=rule_color,show_legend=False,label=label,rule_plot=rules,average_rules=average_rules)
        #except:
            #continue

    ax.plot([0,ax.get_xlim()[1]],[0.95,0.95],'k',linestyle=':')
    #ax.set_xlim([0,ax.get_xlim()[1]])
    if show_legend and labels:
        ax.legend(custom_lines, labels, loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12,labelspacing=0.3,frameon=False)

    plt.show()

def swap_indices(array, ind_pair):
    new_array = array.copy()
    new_array[array == ind_pair[0]] = ind_pair[1]
    new_array[array == ind_pair[1]] = ind_pair[0]

    return new_array


def relabel_cluster(cluster_object, cluster_pair):
    cluster_object.labels = swap_indices(cluster_object.labels, cluster_pair)
    cluster_object.all_labels = swap_indices(cluster_object.all_labels, cluster_pair)
    cluster_object.active_labels = swap_indices(cluster_object.active_labels, cluster_pair)
    cluster_object.scores = swap_indices(cluster_object.scores, cluster_pair)
    cluster_object.unique_labels = swap_indices(cluster_object.unique_labels, cluster_pair)

    return cluster_object



def lesion_weights(weigth_matrix,perc_lesioned_weights,clustered_rnn_geometry): 
    lin_weigth_matrix = np.abs(weigth_matrix.flatten())
    lin_weigth_matrix = np.sort(lin_weigth_matrix)

    lim_weights = lin_weigth_matrix[int(perc_lesioned_weights*len(lin_weigth_matrix))]

    lesion_weights_list = np.array(np.where((weigth_matrix < lim_weights)*(weigth_matrix > -1*lim_weights))).T

    perfs_changes, cost_changes = clustered_rnn_geometry.lesions(lesion_weights_list=lesion_weights_list)

    return perfs_changes



def video_activations(dict_activations, fps, downsample=None,video_name='',zscoring=False):

    if downsample is None:
        downsample = 1

    t_range = range(0,len(dict_activations[list(dict_activations.keys())[0]][1]),downsample)

    fig = plt.figure(figsize=[15, 5])

    ax = []
    plots = []

    for n_plot, key in enumerate(dict_activations):
        unit_pos = dict_activations[key][0]
        z = dict_activations[key][1]
        z = z - np.mean(z, 0)
        if zscoring:
            z = z/np.std(z, 0)

        max_z = np.max(z[:])
        min_z = np.min(z[:])

        x = [unit[0] for unit in unit_pos]
        y = [unit[1] for unit in unit_pos]
        c = z[t_range[1], :]# * 0.7

        ax.append(fig.add_subplot(1, 3, n_plot + 1))

        plots.append(ax[n_plot].scatter(x, y, c=c, s=50, marker='o', vmax=max_z, vmin=min_z, cmap='hot'))
        plots[n_plot].set_edgecolor('k')

        ax[n_plot].grid(False)
        # ax[n_plot].set_axis_off()
        ax[n_plot].set_aspect('equal', adjustable='box')
        ax[n_plot].set_xlim([-0.05, 1.1])
        ax[n_plot].set_ylim([-0.05, 1.1])
        ax[n_plot].set_xticks([])
        ax[n_plot].set_yticks([])

    def update_img(t):
        for n_plot, key in enumerate(dict_activations):
            z = dict_activations[key][1]
            z = z - np.mean(z, 0)
            if zscoring:
                z = z / np.std(z, 0)
            c = z[t_range[t], :]# * 0.7
            cmap_vals = plt.get_cmap('hot')
            c = [cmap_vals(col) for col in c]
            plots[n_plot].set_facecolor(c)
        return plots

    ani = animation.FuncAnimation(fig, update_img, int(len(t_range)), interval=1)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save('video_' + video_name + '.mp4', writer=writer, dpi=100)
    return ani


def get_all_unit_activations(model_dir, rule, params, average_activations=True, ommit_fix_unit=False, zscoring=False,
                             pre_trial_rule='same_rule_same_mod', pre_params=None):
    """
    Args:
        model_dir : model name
        rule : task to analyze
        params_list : a list of parameter dictionaries used for the psychometric mode
        batch_shape : shape of each batch. Each batch should have shape (n_rep, ...)
        n_rep is the number of repetitions that will be averaged over

    Return:
        ydatas: list of performances
    """

    model = Model(model_dir)
    hp = model.hp
    with tf.compat.v1.Session() as sess:
        model.restore()

        trial = task.generate_trials(rule, hp, 'psychometric', params=params)
        h_init = None
        if 'transfer_h_across_trials' in hp and hp['transfer_h_across_trials']:
            batch_size=params['stim1_locs'].shape[0]
            if pre_trial_rule == 'same_rule_same_mod':
                pre_trial = trial
            elif pre_trial_rule == 'same_rule_rand_mod':
                pre_trial = task.generate_trials(rule, hp, 'random', batch_size=batch_size)
            elif pre_trial_rule == 'opp_rule_ran_mod':
                pre_rule = rule[:-1] + str(int(rule[-1])%2+1)
                pre_trial = task.generate_trials(pre_rule, hp, 'random', batch_size=batch_size)
            elif pre_trial_rule == 'opp_rule_same_mod':
                pre_rule = rule[:-1] + str(int(rule[-1]) % 2 + 1)
                pre_trial = task.generate_trials(pre_rule, hp, 'psychometric', params=params)
            feed_dict = tools.gen_feed_dict(model, pre_trial, hp, h_init)
            h_init = sess.run(model.h[-1, :, :], feed_dict=feed_dict)
        feed_dict = tools.gen_feed_dict(model, trial, hp, h_init)
        x = sess.run(model.x, feed_dict=feed_dict)
        h = sess.run(model.h, feed_dict=feed_dict)
        y_hat = sess.run(model.y_hat, feed_dict=feed_dict)

    #model_data = get_model_data(model_dir, get_clusters=False)

    if average_activations:
        x = np.mean(x, 1)
        h = np.mean(h, 1)
        y_hat = np.mean(y_hat, 1)

    if zscoring:
        x = x - np.mean(x, 0)
        x = x / np.std(x, 0)
        h = h - np.mean(h, 0)
        h = h / np.std(h, 0)
        y_hat = y_hat - np.mean(y_hat, 0)
        y_hat = y_hat / np.std(y_hat, 0)

    if ommit_fix_unit:
        dict_activations = {'input': x,
                            'hidden': h,
                            'output': y_hat[:, 1:]}
    else:
        dict_activations = {'input': x,
                            'hidden': h,
                            'output': y_hat}

    return dict_activations


def get_sensory_stim_params(stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, n_repats=100, stim_time=1300, stim1_locs_pi = 0, stim1_locs_idx = 0, single_loc=True):
    n_stim_loc = n_repats  # n of repeats
    n_stim = 1
    batch_size = n_stim_loc * n_stim ** 2
    batch_shape = (n_stim_loc, n_stim, n_stim)
    ind_stim_loc, ind_stim_mod1, ind_stim_mod2 = np.unravel_index(range(batch_size), batch_shape)

    if n_stim_loc > 8:
        if single_loc:
            # Use only one target location now
            stim1_locs = np.zeros(len(ind_stim_loc)) + stim1_locs_pi * np.pi
        else:
            # Looping target location
            stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc

        stim2_locs = (stim1_locs + np.pi) % (2 * np.pi)
    else:
        if single_loc:
            # Use only one target location now
            stim1_locs = np.zeros(len(ind_stim_loc)) + stim1_locs_idx * np.pi
        else:
            # Looping target location
            stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc

        stim2_locs = (stim1_locs + np.pi) % (2 * np.pi)


    params = {'stim1_locs': stim1_locs,
              'stim2_locs': stim2_locs,
              'stim1_mod1_strengths': stim1_mod1[ind_stim_mod1],
              'stim2_mod1_strengths': stim2_mod1[ind_stim_mod1],
              'stim1_mod2_strengths': stim1_mod2[ind_stim_mod2],
              'stim2_mod2_strengths': stim2_mod2[ind_stim_mod2],
              'stim_time': stim_time}

    return params



def input_output_surface_plot(rule,params):

    hp = get_hp()
    trial = task.generate_trials(rule, hp, 'psychometric', params=params)

    fig = plt.figure(figsize=(20,10))
    ax = []

    for plot in range(1,4):
        if plot == 3:
            response_space = trial.x.squeeze()[:,1:33]
            plot_label = 'Input units - Modality 1'
        elif plot == 2:
            response_space = trial.x.squeeze()[:,33:65]
            plot_label = 'Input units - Modality 2'
        elif plot == 1:
            response_space = trial.y.squeeze()[:,1:]
            plot_label = 'Correct output'

        units_mesh, time_mesh = np.meshgrid(range(1,33),range(trial.x.squeeze().shape[0]))
        ax.append(fig.add_subplot(2,2,plot, projection='3d'))
        ax[-1].plot_surface(units_mesh,time_mesh,response_space, cmap=cm.hot)
        ax[-1].set_xlabel(plot_label, labelpad=20)
        ax[-1].set_ylabel('Time', labelpad=20)
        ax[-1].set_xlim([1, 32])

    max_zlim = np.max(np.concatenate([a.get_zlim() for a in ax]))
    ax[0].set_zlim([-0.05,max_zlim])
    ax[1].set_zlim([-0.05,max_zlim])
    ax[2].set_zlim([-0.05,max_zlim])


def plot_weight_matrix(model_data,w_type='hidden',abs_weights=False,fft2=False,hist=False):
    if w_type == 'input':
        weight_matrix = model_data['w_in']
        figsz = [15,8]
    elif w_type == 'output':
        weight_matrix = model_data['w_out'].T
        figsz = [15,8]
    elif w_type == 'hidden':
        weight_matrix = model_data['w_rec']
        figsz = [15,15]

    print(str(round(1 - sum(weight_matrix.flatten() == 0) / len(weight_matrix.flatten()), 2) * 100)
          + '% of weights trained.')

    if abs_weights:
        weight_matrix = np.abs(weight_matrix)#-np.min(weight_matrix.flatten())
        cmap = cm.get_cmap('hot')
    else:
        #weight_matrix = weight_matrix-np.mean(weight_matrix.flatten())
        cmap = cm.get_cmap('coolwarm_r')

    #weight_matrix[weight_matrix == 0] = np.nan
    cmap.set_bad(color=[230/255, 230/255, 230/255]) #(color='white')
    plt.figure(figsize=[15, 8])
    ax = plt.axes()
    plt.imshow(weight_matrix, cmap=cmap)
    max_val = np.max(np.abs(weight_matrix.flatten()))
    plt.clim([-max_val, max_val])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if fft2:
        f = np.fft.fft2(weight_matrix)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        plt.figure(figsize=[15,8])
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.xticks([]), plt.yticks([])

    if hist:
        plt.figure(figsize=[5, 5])
        plt.hist(weight_matrix.flatten()[np.logical_not(np.isnan(weight_matrix.flatten()))], density=True)


def subsample_unit_activations(activations, indexes, with_repeats=False):
    dict_activations_subsample = dict.fromkeys(activations.keys())

    dict_activations_subsample['input'] = activations['input']
    if with_repeats:
        print(activations['hidden'].shape)
        dict_activations_subsample['hidden'] = activations['hidden'][:, :, indexes] #[indexes, :]
    else:
        dict_activations_subsample['hidden'] = activations['hidden'][:, indexes] #[indexes, :]
    dict_activations_subsample['output'] = activations['output']

    return dict_activations_subsample


def get_activations(model_dir, rule,
                    stim1_mod1=np.array([0]), stim2_mod1=np.array([0]),
                    stim1_mod2=np.array([0]), stim2_mod2=np.array([0]),
                    subsample_indexes=None, n_directions=2, pre_trial_rule='same_rule_same_mod',
                    single_direction=None):

    if single_direction is None:
        params = get_sensory_stim_params(stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, n_repats=n_directions,
                                         single_loc=False)
    else:
        if n_directions > 8:
            params = get_sensory_stim_params(stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, n_repats=n_directions,
                                             single_loc=True, stim1_locs_pi=single_direction)
        else:
            params = get_sensory_stim_params(stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, n_repats=n_directions,
                                             single_loc=True, stim1_locs_idx=single_direction)

    print(params)
    dict_activations = get_all_unit_activations(model_dir, rule, params, average_activations=False,
                                                pre_trial_rule=pre_trial_rule)

    if subsample_indexes is not None:
        dict_activations = subsample_unit_activations(dict_activations, subsample_indexes, with_repeats=True)

    temp_activity_neurons = dict_activations['hidden']
    time_steps = temp_activity_neurons.shape[0]
    unique_stim_locs = temp_activity_neurons.shape[1]
    activity_neurons = np.zeros([temp_activity_neurons.shape[2], time_steps * unique_stim_locs])

    for n in range(activity_neurons.shape[0]):
        neuron_activity = np.zeros([time_steps * unique_stim_locs])
        for loc in range(unique_stim_locs):
            neuron_activity[(loc * time_steps):((loc + 1) * time_steps)] = temp_activity_neurons[:, loc, n]
        activity_neurons[n, :] = neuron_activity

    return activity_neurons, unique_stim_locs, time_steps


def get_PC_projections(activity_neurons, unique_stim_locs, time_steps):

    demeaned_activity_neurons = activity_neurons - np.tile(np.mean(activity_neurons, 1),
                                                           [activity_neurons.shape[1], 1]).T

    # plt.imshow(demeaned_activity_neurons)

    def cov_matrix(data_matrix):
        print(np.shape(data_matrix))
        data_matrix_zeromean = data_matrix - np.mean(data_matrix, axis=1, keepdims=True)
        print(np.shape(np.mean(data_matrix, axis=0)))
        Y = np.matmul(data_matrix_zeromean, np.transpose(data_matrix_zeromean)) / (data_matrix_zeromean.shape[1])
        return Y

    Y = cov_matrix(demeaned_activity_neurons)
    # plt.figure(figsize=(5,5))
    # plt.imshow(Y)

    eigVals, eigVects = np.linalg.eig(Y)

    def projections(eigenvec, data_nomean):
        projections = np.dot(data_nomean.T, eigenvec)
        return projections

    return [np.reshape(projections(eigVects[:, i], demeaned_activity_neurons), (unique_stim_locs, time_steps)) for i in range(3)]


def PCAs_single_rule_single_mod_all_sides(model_dir, rule, subnetworks,
                                          stim1_mod1=np.array([0]), stim2_mod1=np.array([0]),
                                          stim1_mod2=np.array([0]), stim2_mod2=np.array([0]),
                                          n_directions=2, pre_trial_rule='same_rule_same_mod'):
    for i in range(len(subnetworks)):
        activity_neurons, unique_stim_locs, time_steps = get_activations(model_dir, rule,
                                                                         stim1_mod1=stim1_mod1,
                                                                         stim1_mod2=stim1_mod2,
                                                                         subsample_indexes=subnetworks[i]['index_range'],
                                                                         n_directions=n_directions, pre_trial_rule=pre_trial_rule)
        subnetworks[i]['pc_projections'] = get_PC_projections(activity_neurons, unique_stim_locs, time_steps)
    return subnetworks


def PCAs_all_rules_single_mods_single_side(model_dir, rules, subnetworks, direction=0, mod=0,
                                           pre_trial_rule='same_rule_same_mod'):

    if mod > 0:
        m = [[1, 0], [0, 1]][mod - 1]
        stim1_mod1 = np.array([m[0]])
        stim1_mod2 = np.array([m[1]])

    for i in range(len(subnetworks)):
        activity_neurons = None
        unique_stim_locs = 0
        for r, rule in enumerate(rules):
            if mod == 0:
                m = [[1, 0], [0, 1]][r]
                stim1_mod1 = np.array([m[0]])
                stim1_mod2 = np.array([m[1]])

            temp_activity_neurons, temp_unique_stim_locs, time_steps = get_activations(model_dir, rule,
                                                                                       stim1_mod1=stim1_mod1,
                                                                                       stim1_mod2=stim1_mod2,
                                                                                       subsample_indexes=subnetworks[i]['index_range'],
                                                                                       n_directions=1,
                                                                                       pre_trial_rule=pre_trial_rule,
                                                                                       single_direction=direction)
            if activity_neurons is None:
                activity_neurons = temp_activity_neurons
            else:
                activity_neurons = np.concatenate([activity_neurons, temp_activity_neurons], axis=1)
            unique_stim_locs += temp_unique_stim_locs

        subnetworks[i]['pc_projections'] = get_PC_projections(activity_neurons, unique_stim_locs, time_steps)

    return subnetworks


def PCAs_single_rule_all_mods_single_side(model_dir, rule, subnetworks, direction=0, pre_trial_rule='same_rule_same_mod'):
    for i in range(len(subnetworks)):
        activity_neurons = None
        unique_stim_locs = 0
        for m, mod in enumerate([[1, 0], [0, 1]]):
            stim1_mod1 = np.array([mod[0]])
            stim1_mod2 = np.array([mod[1]])

            temp_activity_neurons, temp_unique_stim_locs, time_steps = get_activations(model_dir, rule,
                                                                                       stim1_mod1=stim1_mod1,
                                                                                       stim1_mod2=stim1_mod2,
                                                                                       subsample_indexes=
                                                                                       subnetworks[i]['index_range'],
                                                                                       n_directions=1,
                                                                                       pre_trial_rule=pre_trial_rule,
                                                                                       single_direction=direction)
            if activity_neurons is None:
                activity_neurons = temp_activity_neurons
            else:
                activity_neurons = np.concatenate([activity_neurons, temp_activity_neurons], axis=1)
            unique_stim_locs += temp_unique_stim_locs

        subnetworks[i]['pc_projections'] = get_PC_projections(activity_neurons, unique_stim_locs, time_steps)

    return subnetworks


def PCAs_all_rules_all_mods_all_sides(model_dir, rules, subnetworks, n_directions=2, pre_trial_rule='same_rule_same_mod'):
    for i in range(len(subnetworks)):
        activity_neurons = None
        unique_stim_locs = 0
        for r, rule in enumerate(rules):
            for m, mod in enumerate([[1, 0], [0, 1]]):
                stim1_mod1 = np.array([mod[0]])
                stim1_mod2 = np.array([mod[1]])

                temp_activity_neurons, temp_unique_stim_locs, time_steps = get_activations(model_dir, rule,
                                                                                           stim1_mod1=stim1_mod1,
                                                                                           stim1_mod2=stim1_mod2,
                                                                                           subsample_indexes=
                                                                                           subnetworks[i]['index_range'],
                                                                                           n_directions=n_directions,
                                                                                           pre_trial_rule=pre_trial_rule)
                if activity_neurons is None:
                    activity_neurons = temp_activity_neurons
                else:
                    activity_neurons = np.concatenate([activity_neurons, temp_activity_neurons], axis=1)
                unique_stim_locs += temp_unique_stim_locs

        subnetworks[i]['pc_projections'] = get_PC_projections(activity_neurons, unique_stim_locs, time_steps)

    return subnetworks


def plot_PC_projections(projections, plot_single_pc=True, plot_2d_3d='2d', plot_locs=True, given_2d_3d_ax=None,
                        colors=None, title=''):

    #plt.rcParams.update({'font.size': 12})
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    unique_stim_locs = projections[0].shape[0]
    time_steps = projections[0].shape[1]

    axes = []

    if plot_single_pc:
        plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        plt.subplots_adjust(wspace=0.4)
        axes.append(ax1)
        axes.append(ax2)
        axes.append(ax3)
    else:
        if given_2d_3d_ax is None:
            if plot_2d_3d == '2d':
                plt.figure(figsize=(5, 5))
                ax_2d_3d = plt.subplot(121)
            else:
                plt.figure(figsize=(7, 7))
                ax_2d_3d = plt.subplot(131, projection='3d')
        else:
            ax_2d_3d = given_2d_3d_ax

    if plot_locs:
        plt.figure(figsize=(5, 5))
        ax_locs = plt.axes()
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        ax_locs.set_axis_off()

    lines = []
    point_angle = 2 * np.pi

    for loc in range(unique_stim_locs):

        pc1 = projections[0][loc,:]
        pc2 = projections[1][loc,:]
        pc3 = projections[2][loc,:]

        if plot_single_pc:
            ax1.plot(pc1, color=colors[loc])
            ax2.plot(pc2, color=colors[loc])
            ax3.plot(pc3, color=colors[loc])
        else:
            if plot_2d_3d == '2d':
                lines.append(ax_2d_3d.plot(pc1, pc2, color=colors[loc]))
            else:
                lines.append(ax_2d_3d.plot3D(pc1, pc2, pc3, color=colors[loc]))

        if plot_locs:
            rad = 1
            col = lines[-1][-1].get_color()

            for loc_point in range(int(32 / unique_stim_locs)):
                point_num = (loc + 1) * (loc_point + 1)
                point_angle -= 2 * np.pi / 32
                ax_locs.scatter(np.cos(point_angle), np.sin(point_angle), c=col, s=100)

    if plot_single_pc:
        ax1.set_title(title + ' (PC1)')
        ax2.set_title(title + ' (PC2)')
        ax3.set_title(title + ' (PC3)')


def plot_subnetwork_PC_projections(subnetworks, plot_2d_3d='2d', plot_single_pc=False, title='', colors=None):
    if plot_single_pc:
        plt.figure(figsize=(10, 10))
        for i in range(len(subnetworks)):
            plot_PC_projections(subnetworks[i]['pc_projections'], plot_single_pc=True, plot_locs=False,
                                colors=colors, title=subnetworks[i]['name'] + title)
    else:
        if plot_2d_3d == '2d':
            plt.figure(figsize=(10, 10))
        else:
            plt.figure(figsize=(20, 10))
        subplots = []
        global_x_lims = [0, 0]
        global_y_lims = [0, 0]
        global_z_lims = [0, 0]

        for i in range(len(subnetworks)):
            if plot_2d_3d=='2d':
                subplots.append(plt.subplot(231 + i))
            else:
                subplots.append(plt.subplot(231 + i, projection='3d'))
            plot_PC_projections(subnetworks[i]['pc_projections'], plot_single_pc=False, plot_locs=False,
                                     given_2d_3d_ax=subplots[-1], plot_2d_3d=plot_2d_3d, colors=colors)

            x_lims = subplots[i].get_xlim()
            y_lims = subplots[i].get_ylim()
            if x_lims[0] < global_x_lims[0]:
                global_x_lims[0] = x_lims[0]
            if x_lims[1] > global_x_lims[1]:
                global_x_lims[1] = x_lims[1]
            if y_lims[0] < global_y_lims[0]:
                global_y_lims[0] = y_lims[0]
            if y_lims[1] > global_y_lims[1]:
                global_y_lims[1] = y_lims[1]

            if plot_2d_3d == '3d':
                z_lims = subplots[i].get_zlim()
                if z_lims[0] < global_z_lims[0]:
                    global_z_lims[0] = z_lims[0]
                if z_lims[1] > global_z_lims[1]:
                    global_z_lims[1] = x_lims[1]

            plt.title(subnetworks[i]['name'] + title)

        for i in range(len(subnetworks)):
            subplots[i].set_xlim(global_x_lims)
            subplots[i].set_ylim(global_y_lims)
            if plot_2d_3d == '3d':
                subplots[i].set_zlim(global_z_lims)

            if i < 3:
                subplots[i].axes.get_xaxis().set_visible(False)
            if i != 0 and i != 3:
                subplots[i].axes.get_yaxis().set_visible(False)


def plot_unit_activations(activations, global_x_lims=[0, 0], global_y_lims=[0, 0], color='grey', given_subplots=None, subplot_shapes=None):
    n = activations.shape[2]
    if subplot_shapes is None:
        side1 = int(np.ceil(np.sqrt(n)))
        side2 = int(np.ceil(n / side1))
    else:
        side1 = subplot_shapes[0]
        side2 = subplot_shapes[1]

    plt.figure(figsize=[3*side2, 3*side1])

    subplots = []

    for i in range(n):
        if given_subplots is None:
            subplots.append(plt.subplot(side1, side2, i + 1))
        else:
            subplots.append(given_subplots[i])
        ax = subplots[-1]
        ax.plot(activations[:, :, i], color, alpha=0.2)
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        if x_lims[0] < global_x_lims[0]:
            global_x_lims[0] = x_lims[0]
        if x_lims[1] > global_x_lims[1]:
            global_x_lims[1] = x_lims[1]
        if y_lims[0] < global_y_lims[0]:
            global_y_lims[0] = y_lims[0]
        if y_lims[1] > global_y_lims[1]:
            global_y_lims[1] = y_lims[1]
    # print(global_y_lims)
    for i in range(len(subplots)):
        subplots[i].set_xlim(global_x_lims)
        subplots[i].set_ylim(global_y_lims)
        if i < (side1-1)*side2:
            subplots[i].axes.get_xaxis().set_visible(False)
        if i%side2:
            subplots[i].axes.get_yaxis().set_visible(False)

    return subplots, global_x_lims, global_y_lims


def compare_unit_activations(list_activations, colors, subplot_shapes=None):
    given_subplots = None
    global_x_lims = [0, 0]
    global_y_lims = [0, 0]
    for i in range(len(list_activations)):
        given_subplots, global_x_lims, global_y_lims = plot_unit_activations(list_activations[i], global_x_lims, global_y_lims,
                                                                             color=colors[i], given_subplots=given_subplots,
                                                                             subplot_shapes=subplot_shapes)


def perf_pre_post_lesion(model_dir, rule, units_to_lesion, pre_trial_rule='same_rule_same_mod'):

    model = Model(model_dir)
    hp = model.hp
    with tf.compat.v1.Session() as sess:
        model.restore()

        trial = task.generate_trials(rule, hp, 'random', batch_size=1000)
        h_init = None
        if 'transfer_h_across_trials' in hp and hp['transfer_h_across_trials']:
            if pre_trial_rule == 'same_rule_same_mod':
                pre_trial = trial
            elif pre_trial_rule == 'same_rule_rand_mod':
                pre_trial = task.generate_trials(rule, hp, 'random', batch_size=1000)
            elif pre_trial_rule == 'opp_rule_ran_mod':
                pre_rule = rule[:-1] + str(int(rule[-1])%2+1)
                pre_trial = task.generate_trials(pre_rule, hp, 'random', batch_size=1000)
            elif pre_trial_rule == 'opp_rule_same_mod':
                pre_rule = rule[:-1] + str(int(rule[-1])%2+1)
                trial_rule_x = trial.x[:, :, task.get_rule_index(rule, trial.config)]
                trial_pre_rule_x = trial.x[:, :, task.get_rule_index(pre_rule, trial.config)]
                pre_trial = copy.deepcopy(trial)
                pre_trial.x[:, :, task.get_rule_index(rule, trial.config)] = trial_pre_rule_x
                pre_trial.x[:, :, task.get_rule_index(pre_rule, trial.config)] = trial_rule_x
            feed_dict = tools.gen_feed_dict(model, pre_trial, hp, h_init)
            h_init = sess.run(model.h[-1, :, :], feed_dict=feed_dict)
        feed_dict = tools.gen_feed_dict(model, trial, hp, h_init)
        #x = sess.run(model.x, feed_dict=feed_dict)
        #h = sess.run(model.h, feed_dict=feed_dict)
        y_hat = sess.run(model.y_hat, feed_dict=feed_dict)

        perf_pre = get_perf(y_hat, trial.y_loc)

        if bool(units_to_lesion):
            model.lesion_units(sess, units_to_lesion)

        y_hat = sess.run(model.y_hat, feed_dict=feed_dict)
        perf_post = get_perf(y_hat, trial.y_loc)

    # plt.figure()
    # plt.plot(trial.x[:, :, task.get_rule_index(rule, trial.config)])
    # plt.figure()
    # plt.plot(pre_trial.x[:, :, task.get_rule_index(rule, trial.config)])
    return sum(perf_pre)/len(perf_pre), sum(perf_post)/len(perf_post)


def bar_plot(data_points, labels=None, colors=None, title=None):

    if colors is None:
        colors = ['k' for _ in data_points]

    plt.figure(figsize=(3, 4))

    ax = plt.axes()
    bar_pos = np.arange(0,2*len(data_points),2)
    plt.bar(bar_pos, [dp * 100 for dp in data_points], color=colors)

    ax.set_xticks(bar_pos)
    if labels is None:
        ax.set_xticklabels(['Ctrl', 'Lesion'])
    else:
        ax.set_xticklabels(labels)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([ax.get_xlim()[0]-1, ax.get_xlim()[1]+1])

    plt.ylabel('Performance %')

    if title is not None:
        plt.title(title)

    plt.show()

