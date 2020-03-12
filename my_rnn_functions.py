from os.path import join
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


def plot_performanceprogress(model_dir, rule_color, ax=None, fig=None, rule_plot=None, label=None, show_legend=False):
    # Plot Training Progress
    log = tools.load_log(model_dir)
    hp = tools.load_hp(model_dir)

    trials = log['trials']

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
        line = ax.plot(x_plot, np.log10(log['cost_'+rule]),
                       color=rule_color[rule])
        ax.plot(x_plot, log['perf_'+rule], color=rule_color[rule])
        lines.append(line[0])
        if label:
            labels.append(label)
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
    


def plt_various_performances(trained_models,models_saving_dir='./saved_models',rules=['multidelaydm','contextdm1','contextdm2'], colors_to_use = 'tableau', show_legend=False, labels=None):


    if colors_to_use == 'tableau':
        colors = mcolors.TABLEAU_COLORS
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        rule_color_pairs = [name for hsv, name in by_hsv]
    else:
        rule_color_pairs = ['bright purple','green blue','indigo','grey blue','lavender','aqua']

    fig_all = plt.figure(figsize=(15,6))

    custom_lines = []

    for i, trained_model in enumerate(trained_models):
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
        _, ax = plot_performanceprogress(model_dir, fig=fig_all , ax=ax, rule_color=rule_color,show_legend=False,label=label,rule_plot=rules)

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
                             pre_trial_rule='opp_random', pre_params=None):
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
            if pre_trial_rule == 'same':
                pre_trial = trial
            elif pre_trial_rule == 'same_rand':
                pre_trial = task.generate_trials(rule, hp, 'random', batch_size=batch_size)
            elif pre_trial_rule == 'opp_random':
                pre_rule = rule[:-1] + str(int(rule[-1])%2+1)
                pre_trial = task.generate_trials(pre_rule, hp, 'random', batch_size=batch_size)
            elif pre_trial_rule == 'opp_random':
                pre_trial = task.generate_trials(rule, hp, 'psychometric', params=pre_params)
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


def get_sensory_stim_params(stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, n_repats=100, stim_time=800, stim1_locs_pi = 0, single_loc=True):
    n_stim_loc = n_repats  # n of repeats
    n_stim = 1
    batch_size = n_stim_loc * n_stim ** 2
    batch_shape = (n_stim_loc, n_stim, n_stim)
    ind_stim_loc, ind_stim_mod1, ind_stim_mod2 = np.unravel_index(range(batch_size), batch_shape)

    if single_loc:
        # Use only one target location now
        stim1_locs = np.zeros(len(ind_stim_loc)) + stim1_locs_pi * np.pi
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


def PCA_activity(model_dir, rule, stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, subsample_indexes=None):

    params = get_sensory_stim_params(stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, n_repats=8, single_loc=False)

    dict_activations = get_all_unit_activations(model_dir, rule, params, average_activations=False)

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

    demeaned_activity_neurons = activity_neurons - np.tile(np.mean(activity_neurons, 1),
                                                           [activity_neurons.shape[1], 1]).T
    # plt.imshow(demeaned_activity_neurons)

    for n in range(activity_neurons.shape[0]):
        neuron_activity = np.zeros([time_steps * unique_stim_locs])
        for loc in range(unique_stim_locs):
            neuron_activity[(loc * time_steps):((loc + 1) * time_steps)] = temp_activity_neurons[:, loc, n]
        activity_neurons[n, :] = neuron_activity

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


def PCAs_on_subnetworks(model_dir, rule, stim1_mod1, stim2_mod1, stim1_mod2, stim2_mod2, subnetworks):
    for i in range(len(subnetworks)):
        subnetworks[i]['pc_projections'] = PCA_activity(model_dir, rule, stim1_mod1, stim2_mod1, stim1_mod2,
                                                             stim2_mod2,
                                                             subsample_indexes=subnetworks[i]['index_range'])
    return subnetworks


def plot_PC_projections(projections, plot_single_pc=True, plot_2d_3d='2d', plot_locs=True, given_2d_3d_ax=None):

    #plt.rcParams.update({'font.size': 12})

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
            ax1.plot(pc1)
            ax2.plot(pc2)
            ax3.plot(pc3)

        if plot_2d_3d == '2d':
            lines.append(ax_2d_3d.plot(pc1, pc2))
        else:
            lines.append(ax_2d_3d.plot3D(pc1, pc2, pc3))

        if plot_locs:
            rad = 1
            col = lines[-1][-1].get_color()

            for loc_point in range(int(32 / unique_stim_locs)):
                point_num = (loc + 1) * (loc_point + 1)
                point_angle -= 2 * np.pi / 32
                ax_locs.scatter(np.cos(point_angle), np.sin(point_angle), c=col, s=100)


def plot_subnetwork_PC_projections(subnetworks):
    plt.figure(figsize=(10, 10))
    subplots = []
    global_x_lims = [0, 0]
    global_y_lims = [0, 0]

    for i in range(len(subnetworks)):
        subplots.append(plt.subplot(231 + i))
        plot_PC_projections(subnetworks[i]['pc_projections'], plot_single_pc=False, plot_locs=False,
                                 given_2d_3d_ax=subplots[-1])
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
        plt.title(subnetworks[i]['name'])

    for i in range(len(subnetworks)):
        subplots[i].set_xlim(global_x_lims)
        subplots[i].set_ylim(global_y_lims)
        if i < 3:
            subplots[i].axes.get_xaxis().set_visible(False)
        if i != 0 and i != 3:
            subplots[i].axes.get_yaxis().set_visible(False)


def plot_unit_activations(activations, global_x_lims=[0, 0], global_y_lims=[0, 0]):
    n = activations.shape[2]
    side1 = int(np.ceil(np.sqrt(n)))
    side2 = int(np.ceil(n / side1))

    plt.figure(figsize=[50, 50])

    subplots = []

    for i in range(n):
        subplots.append(plt.subplot(side1, side2, i + 1))
        plt.plot(activations[:, :, i], 'grey', alpha=0.2)
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

    for i in range(len(subplots)):
        subplots[i].set_xlim(global_x_lims)
        subplots[i].set_ylim(global_y_lims)
    #     if i < 3:
    #         subplots[i].axes.get_xaxis().set_visible(False)
    #     if i != 0 and i != 3:
    #         subplots[i].axes.get_yaxis().set_visible(False)

    print(global_y_lims)


def perf_pre_post_lesion(model_dir, rule, units_to_lesion, pre_trial_rule='opp_random'):
    model = Model(model_dir)
    hp = model.hp
    with tf.compat.v1.Session() as sess:
        model.restore()

        trial = task.generate_trials(rule, hp, 'random', batch_size=1000)
        h_init = None
        if 'transfer_h_across_trials' in hp and hp['transfer_h_across_trials']:
            if pre_trial_rule == 'same':
                pre_trial = trial
            elif pre_trial_rule == 'same_rand':
                pre_trial = task.generate_trials(rule, hp, 'random', batch_size=1000)
            elif pre_trial_rule == 'opp_random':
                pre_rule = rule[:-1] + str(int(rule[-1])%2+1)
                pre_trial = task.generate_trials(pre_rule, hp, 'random', batch_size=1000)
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

    return sum(perf_pre)/len(perf_pre), sum(perf_post)/len(perf_post)