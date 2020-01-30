import tensorflow as tf

import numpy as np

class SimpleRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", layer_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = tf.keras.layers.SimpleRNNCell(units,
                                                             activation=None)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.activations.get(activation)

    #@tf.function
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        if self.layer_norm:
            outputs = self.activation(self.layer_norm(outputs))
        else:
            outputs = self.activation(outputs)
        return outputs, [outputs]


class TC_module(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, rnn_cell, n_rnn_units, rnn_activation, n_ff_pre_units, context_input_size, output_to_center_size,
                 center_size, ff_activation="relu", **kwargs):
        super().__init__(dynamic=True,**kwargs)
        #self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.context_input_size = context_input_size
        self.output_to_center_size = output_to_center_size
        self.total_output_size = output_to_center_size + output_size if output_size else output_to_center_size
        self.n_ff_pre_units = n_ff_pre_units
        self.ff_activation = ff_activation
        self.n_rnn_units = n_rnn_units
        self.rnn_activation = rnn_activation
        self.rnn_cell = rnn_cell
        self.center_size = center_size
        self.module_input_size = module_input_size = input_size[2]+context_input_size if input_size is not None else context_input_size

    def build(self, input_shape):
        self.ff_pre = tf.keras.layers.Dense(self.n_ff_pre_units, activation=self.ff_activation,
                                            input_shape=[1, self.module_input_size])

        self.rnn_cell = self.rnn_cell(self.n_rnn_units, activation=self.rnn_activation, input_shape=[1, self.n_ff_pre_units])

        self.ff_post = tf.keras.layers.Dense(self.total_output_size, activation=self.ff_activation, # consider sigmoid
                                             input_shape=[1, self.n_rnn_units])

        self.reading_weights = tf.Variable(
            tf.random.truncated_normal(shape=[self.center_size, self.context_input_size], stddev=0.1),
            name='reading_weights')
        super().build(input_shape)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        return (tf.zeros([batch_size, 1, self.output_to_center_size], dtype=dtype), # NOTE: consider removing size 1 dim
                tf.zeros([batch_size, 1, self.n_rnn_units], dtype=dtype)) # [module_output_to_center, module_state]

    #@tf.function
    def call(self, inputs, center_state, module_state):
        context_input = tf.matmul(center_state, tf.clip_by_norm(self.reading_weights, 1.0))
        if inputs is not None and len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs,1)

        #module_inputs = tf.concat([inputs, context_input], axis=2) if self.input_size is not None else context_input
        module_inputs = tf.keras.layers.concatenate([inputs, context_input], axis=2) if self.input_size is not None else context_input

        rnn_inputs = self.ff_pre(module_inputs)

        rnn_output, new_module_state = self.rnn_cell(inputs=rnn_inputs, states=module_state)

        module_output = self.ff_post(rnn_output)

        network_output, module_output_to_center = tf.split(module_output,
                                                           [self.output_size, self.output_to_center_size],
                                                           axis=2) if self.output_size is not None else (None, module_output)

        return network_output, module_output_to_center, new_module_state


class ThalNet(tf.keras.models.Model):
    def __init__(self, module_rnn_cell, inputs_sizes, outputs_sizes, context_input_size=30,
                 output_to_center_size=34, rnn_units_per_module=40, ff_pre_units_per_module=20,
                 n_modules=4, rnn_activation="relu", **kwargs):

        super().__init__(**kwargs)
        #total_inputs_sizes = np.sum([i_sz[2] for i_sz in inputs_sizes if i_sz is not None])
        #super().__init__(input_shape=[None,total_inputs_sizes], dynamic=True, **kwargs)

        self._inputs_sizes = inputs_sizes  #[Batch, Time, Num_units]
        #self._n_steps = valid_inputs_sizes[0]
        #self._batch_size = valid_inputs_sizes[0]
        self._outputs_sizes = outputs_sizes
        self._rnn_units_per_module = rnn_units_per_module
        self._ff_pre_units_per_module = ff_pre_units_per_module
        self._context_input_size = context_input_size
        self._center_size = n_modules * output_to_center_size
        self._output_to_center_size = output_to_center_size
        self._n_modules = n_modules

        input_indexes = []
        idx = 0
        for i_sz in inputs_sizes:
            if i_sz is None:
                input_indexes.append(None)
            else:
                input_indexes.append([idx, idx + i_sz[2]])
                idx += i_sz[2]
        self._input_indexes = input_indexes

        self.modules = [TC_module(input_size=self._inputs_sizes[i],
                                  output_size=self._outputs_sizes[i],
                                  rnn_cell=module_rnn_cell,
                                  n_rnn_units=self._rnn_units_per_module,
                                  rnn_activation=rnn_activation,
                                  n_ff_pre_units=self._ff_pre_units_per_module,
                                  context_input_size=self._context_input_size,
                                  center_size=self._center_size,
                                  output_to_center_size=self._output_to_center_size,
                                  name=f'module{i}',
                                  input_shape=[1, self._inputs_sizes[2]] if self._inputs_sizes is not None else None) # NOTE: consider removing size 1 dim
                        for i in range(self._n_modules)]

        self.get_initial_state = [getattr(
            self.modules[m], "get_initial_state", self.fallback_initial_state) for m in range(self._n_modules)]

    def fallback_initial_state(self, inputs):
        return (tf.zeros([1, self._output_to_center_size], dtype=inputs.dtype), # NOTE: consider removing size 1 dim
                tf.zeros([1, self._rnn_units_per_module], dtype=inputs.dtype)) # [module_output_to_center, module_state]

    #@tf.function
    def call(self, inputs):

        n_steps = inputs.shape[1]# if inputs.shape[1] else 1

        center_states, module_states = map(list, zip(*[self.get_initial_state[m](inputs) for m in range(self._n_modules)]))

        #center_states = [tf.zeros([batch_size, self._output_to_center_size], dtype=inputs.dtype) for _ in range(self._n_modules)]
        #module_states = [tf.zeros([batch_size, 1, self._rnn_units_per_module], dtype=inputs.dtype) for _ in range(self._n_modules)]

        temp_outputs = [None for _ in range(self._n_modules)]

        # module_states = [[tf.zeros([1, self._rnn_units_per_module], dtype=inputs.dtype) for b in range(self._batch_size)] for m in range(self._n_modules)]
        # temp_outputs = [tf.zeros(shape=[self._batch_size, self._outputs_sizes[m]], dtype=inputs.dtype)
        #                 if self._outputs_sizes[m] is not None else None
        #                 for m in range(self._n_modules)]

        outputs_list = [tf.TensorArray(inputs.dtype, element_shape=[None, self._outputs_sizes[m]], size=1, dynamic_size=True)#, flow=None if n_steps else tf.TensorArray(inputs.dtype, element_shape=[None, self._outputs_sizes[m]], size=1).flow)
                        if self._outputs_sizes[m] is not None else None
                        for m in range(self._n_modules)]

        if n_steps:
            for step in tf.range(n_steps):

                full_center_state = tf.concat(center_states, axis=2)

                for m in range(self._n_modules):
                    #inputs: [Batch, Time, Num_units]
                    temp_outputs[m], center_states[m], module_states[m] = self.modules[m](
                        inputs[:, step, self._input_indexes[m][0]:self._input_indexes[m][1]] if self._input_indexes[m] is not None else None,
                        full_center_state,
                        module_states[m])
                    if temp_outputs[m] is not None:
                        outputs_list[m] = outputs_list[m].write(step,tf.squeeze(temp_outputs[m]))

        #print([output.stack() for output in outputs_list if output is not None])
        network_outputs = tf.transpose(tf.concat([output.stack() for output in outputs_list if output is not None], axis=2),(1,0,2))

        return network_outputs
