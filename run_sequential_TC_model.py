import tensorflow as tf
import numpy as np
import time
import tasks
from tasks import generate_trials


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
            'n_rnn': 128,
            # number of recurrent units for context and go inputs
            'n_rnn_contexts': 32,
            # number of dense units for output layers of rnn modules
            'n_dense_CT': 16,
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



input_mod1 = tf.keras.Input((None, hp['n_eachring']), name='input_mod1')
input_mod2 = tf.keras.Input((None, hp['n_eachring']), name='input_mod2')
input_context_and_go = tf.keras.Input((None, hp['n_rule']+1), name='input_context_and_go')

rnn_mod1 = tf.keras.layers.SimpleRNN(hp['n_rnn'], return_sequences=True, name='rnn_mod1')(input_mod1)
dense_mod1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp['n_dense_CT']), name='dense_mod1')(rnn_mod1)

rnn_mod2 = tf.keras.layers.SimpleRNN(hp['n_rnn'], return_sequences=True, name='rnn_mod2')(input_mod2)
dense_mod2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp['n_dense_CT']), name='dense_mod2')(rnn_mod2)

rnn_context_and_go = tf.keras.layers.SimpleRNN(hp['n_rnn_contexts'], 
                                               return_sequences=True, 
                                               name='rnn_context_and_go')(input_context_and_go)
dense_context_and_go = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp['n_dense_CT']), 
                                                       name='dense_context_and_go')(rnn_context_and_go)

thal_hub = tf.keras.layers.Concatenate(name='thalamic_hub')([dense_mod1,dense_mod2,dense_context_and_go])

rnn_motor = tf.keras.layers.SimpleRNN(hp['n_rnn'], return_sequences=True, name='rnn_motor')(thal_hub)

output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp['n_output']), name='outputs')(rnn_motor)

TC_model = tf.keras.models.Model(inputs=[input_mod1, input_mod2, input_context_and_go], outputs=output)

TC_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])



max_steps=1e6
step = 0
while step * hp['batch_size_train'] <= max_steps:
    try:
        # Training
        rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                          p=hp['rule_probs'])
        # Generate a random batch of trials.
        # Each batch has the same trial length
        trial = generate_trials(
            rule_train_now, hp, 'random',
            batch_size=hp['batch_size_train'])

        all_inputs = np.transpose(trial.x,(1,0,2))
        inputs_mod1 = all_inputs[:,:,1:(1+hp['n_eachring'])]
        inputs_mod2 = all_inputs[:,:,(1+hp['n_eachring']):(1+2*hp['n_eachring'])]
        inputs_context_and_go = np.concatenate([all_inputs[:,:,:1],all_inputs[:,:,(1+2*hp['n_eachring']):]],axis=2)
        
        targets = np.transpose(trial.y,(1,0,2))
        history = TC_model.fit([inputs_mod1,inputs_mod2,inputs_context_and_go], targets, epochs=20)

        # Generating feed_dict.
        # feed_dict = tools.gen_feed_dict(model, trial, hp)
        # sess.run(model.train_step, feed_dict=feed_dict)

        step += 1

    except KeyboardInterrupt:
        print("Optimization interrupted by user")
        break

print("Optimization finished!")

model_name = "my_TC_model_" + time.strftime("%Y%m%d-%H%M%S")
TC_model.save("./saved_models/" + model_name + ".h5")
TC_model.save_weights('./saved_models/checkpoints/' + model_name)

