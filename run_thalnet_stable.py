import tensorflow as tf

import numpy as np

from thalnet_stable import SimpleRNNCell, TC_module, ThalNet
import tasks
from tasks import generate_trials
import tools

def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    num_ring = tasks.get_num_ring(ruleset)
    n_rule = tasks.get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 512,
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l2 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # Stopping performance
            'target_perf': 0.9,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of modules
            'n_modules': num_ring + 2,
            # number of recurrent units per ring
            'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0,
            'ksi_intsyn': 0,
            }

    return hp


ruleset = 'ctx_multi_sensory_delay'
hp = get_default_hp(ruleset)

seed = 42
hp['seed'] = seed
hp['rng'] = np.random.RandomState(seed)

# Rules to train and test. Rules in a set are trained together

hp['rule_trains'] = tasks.rules_dict[ruleset]
hp['rules'] = hp['rule_trains']

# Assign probabilities for rule_trains.
rule_prob_map = dict()

# Turn into rule_trains format
hp['rule_probs'] = None
if hasattr(hp['rule_trains'], '__iter__'):
    # Set default as 1.
    rule_prob = np.array(
            [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
    hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))


tdim = None
batch_size = hp['batch_size_train']

inputs_sizes = [[tdim,batch_size,hp['n_eachring']+1],  #[Time, Batch, Num_units]
                [tdim,batch_size,hp['n_eachring']],
                [tdim,batch_size,hp['n_rule']],
                None]

outputs_sizes = [None,
                 None,
                 None,
                 hp['n_output']] #[Num_units]

total_input_size = np.sum([sz[2] for sz in inputs_sizes if sz is not None])

tf.keras.backend.clear_session()
thalnet_network = ThalNet(SimpleRNNCell, inputs_sizes, outputs_sizes, n_modules=hp['n_modules'])

#thalnet_network = tf.keras.models.Sequential([ThalNet(SimpleRNNCell, inputs_sizes, outputs_sizes, n_modules=hp['n_modules'], input_shape=[None, batch_size, total_input_size])])

thalnet_network.compile(loss="mse", optimizer="adam")

max_steps=1e6 #1e6
step = 0
print('Starting training...')
while step * hp['batch_size_train'] <= max_steps:
    try:
        # Validation
        # if step % display_step == 0:
        #     log['trials'].append(step * hp['batch_size_train'])
        #     log['times'].append(time.time() - t_start)
        #     log = do_eval(sess, model, log, hp['rule_trains'])
        #     # if log['perf_avg'][-1] > model.hp['target_perf']:
        #     # check if minimum performance is above target
        #     if log['perf_min'][-1] > model.hp['target_perf']:
        #         print('Perf reached the target: {:0.2f}'.format(
        #             hp['target_perf']))
        #         break
        #
        #     if rich_output:
        #         display_rich_output(model, sess, step, log, model_dir)

        # Training
        rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                          p=hp['rule_probs'])
        # Generate a random batch of trials.
        # Each batch has the same trial length
        trial = generate_trials(
            rule_train_now, hp, 'random',
            batch_size=hp['batch_size_train'])

        history = thalnet_network.fit(trial.x, trial.y, epochs=1)

        # Generating feed_dict.
        # feed_dict = tools.gen_feed_dict(model, trial, hp)
        # sess.run(model.train_step, feed_dict=feed_dict)

        step += 1

    except KeyboardInterrupt:
        print("Optimization interrupted by user")
        break

print("Optimization finished!")
