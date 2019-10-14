import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers

"no"

class CTR_ggnn(object):
    def __init__(self, config, device, loader, mode):
        self.config = config
        self.mode = mode
        if mode == "Train":
            self.is_training = False
            self.batch_size = self.config.train_batch_size
            self.maxstep_size = self.config.train_step_size
            reuse = None
        elif mode == "Valid":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            self.maxstep_size = self.config.valid_step_size
            reuse = True
        else:
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            self.maxstep_size = self.config.test_step_size
            reuse = True

        self.hidden_size = hidden_size = config.hidden_size
        self.GNN_step = GNN_step = config.GNN_step
        # self.learning_rate = learning_rate = config.learning_rate
        opt = config.sgd_opt
        beta = config.beta
        batch_size = self.batch_size

        hidden_stdv = np.sqrt(1. / (hidden_size))
        # embedding initial
        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):
            feature_embedding_list = []
            for idx, each_feature_num in enumerate(config.feature_cat_num):
                feature_embedding_list.append(

                    tf.get_variable(name='feature_embedding_' + str(idx), shape=[each_feature_num, hidden_size],
                                    initializer=tf.random_normal_initializer(hidden_stdv))
                )

            w_attention = tf.get_variable(
                name='w_attetion',
                shape=[hidden_size * len(config.feature_num), len(config.feature_num)],
                initializer=tf.random_normal_initializer(stddev=0.2)
            )
            w_score2 = tf.get_variable(
                name='w_score_2', shape=[hidden_size, 1],
                initializer=tf.random_normal_initializer(hidden_stdv)
            )
            float_feature_embedding = tf.get_variable(
                name='float_feature_embedding', shape=[config.feature_float_num, hidden_size],
                initializer=tf.random_normal_initializer(hidden_stdv)
            )

        # #------------feed-----------------##
        print len(config.feature_cat_num)
        print config.feature_float_num
        self.input_x = input_x = tf.placeholder(tf.int32, [batch_size, len(config.feature_cat_num)])
        self.input_x_float = input_x_float = tf.placeholder(tf.float32, [batch_size, config.feature_float_num])

        self.input_y = input_y = tf.placeholder(tf.int32, [batch_size, 1])

        # #--------init graph---------##
        self.graph = graph = init_graph(config, loader)

        #
        # input_x = input_x.transpose((1, 0))
        input_x_unstack = tf.unstack(tf.transpose(input_x, (1, 0)), axis=0)
        input_x_float_unstack = tf.unstack(tf.transpose(input_x_float, (1, 0)), axis=0)

        feature_embedding_input = []
        # translate into embedding (lookup)
        for idx in range(len(input_x_unstack)):
            # if config.feature_flag[idx]:
            feature_embedding_input.append(
                tf.nn.embedding_lookup(feature_embedding_list[idx], input_x_unstack[idx])
            )
        print len(feature_embedding_input)
        for idx in range(len(input_x_float_unstack)):
            # print tf.nn.embedding_lookup(float_feature_embedding, idx).shape
            # print input_x_float_unstack[idx].shape
            float_feature_embedding_idx = tf.nn.embedding_lookup(float_feature_embedding, idx)
            stack_batch_float_feature_embedding_idx = tf.stack(batch_size* [float_feature_embedding_idx])
            # print stack_batch_float_feature_embedding_idx.shape
            # print input_x_float_unstack[idx]
            feature_embedding_input.append(
                stack_batch_float_feature_embedding_idx * tf.reshape(input_x_float_unstack[idx], (batch_size, 1))
            )
        print len(feature_embedding_input)

        self.feature_input= tf.transpose(tf.stack(feature_embedding_input, axis=0), (1, 0 ,2))
        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):
            final_state, test1 = self.GNN(
                self.feature_input, batch_size, hidden_size, GNN_step, len(config.feature_num), graph
            )  # output: [batch_size, config.feature_num, hiddensize]

            atten_pos = self.attention_layer(
                final_state, w_attention, batch_size, hidden_size, len(config.feature_num)
            )
            score_pos = tf.matmul(tf.reshape(final_state, [-1, hidden_size]), w_score2)
            score_pos = tf.maximum(0.01 * score_pos, score_pos)
            score_pos = tf.reshape(score_pos, [batch_size, len(config.feature_num)])
            s_pos = tf.reshape(tf.reduce_sum(score_pos * atten_pos, axis=1), [batch_size, 1])
            # s_pos = self.dense_layer(final_state, batch_size, hidden_size, len(config.feature_num))
            s_pos = tf.reshape(s_pos, [batch_size, 1])
            self.predict = predict = tf.sigmoid(s_pos)
        # -------------evaluation--------------
        self.auc_result, self.auc_opt = tf.metrics.auc(
            labels=self.input_y,
            predictions= predict
        )
        self.s_pos = s_pos
        # -------------cost ---------------
        cost_parameter = 0.
        num_parameter = 0.
        for variable in tf.trainable_variables():
            # print (variable)
            cost_parameter += tf.contrib.layers.l2_regularizer(beta)(variable)
            num_parameter += 1.
        cost_parameter /= num_parameter
        # score = tf.nn.sigmoid(s_pos)
        # score_mean = tf.reduce_mean(score)
        score_mean = tf.losses.log_loss(
            labels=self.input_y,
            predictions=predict
        )
        self.cost = cost = score_mean # + cost_parameter

        # ---------------optimizer---------------#
        self.no_opt = tf.no_op()
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False)

        if mode == 'Train':
            self.auc_opt = tf.no_op()
            self.auc_result = tf.no_op()
            if opt == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost + cost_parameter)
            if opt == 'Momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(cost + cost_parameter)
            if opt == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost + cost_parameter)
            if opt == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cost + cost_parameter)

        else:
            self.optimizer = tf.no_op()

    def weights(self, name, hidden_size, i):
        image_stdv = np.sqrt(1. / (2048))
        hidden_stdv = np.sqrt(1. / (hidden_size))
        if name == 'in_image':
            w = tf.get_variable(name='w/in_image_'+ str(i),
                                shape=[2048, hidden_size],
                                initializer=tf.random_normal_initializer(stddev=image_stdv))
            #w = tf.get_variable(name='gnn/w/in_image_', shape=[2048, hidden_size], initializer=tf.random_normal_initializer)
        if name == 'out_image':
            w = tf.get_variable(name='w/out_image_' + str(i),
                                shape=[hidden_size, 2048],
                                initializer=tf.random_normal_initializer(stddev=image_stdv))
        if name == 'hidden_state':
            if i > 0:
                with tf.variable_scope("w", reuse=True):
                    w = tf.get_variable(name='hidden_state',
                                        shape=[hidden_size, hidden_size],
                                        # initializer=tf.random_normal_initializer(stddev=hidden_stdv)
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name='hidden_state',
                                        shape=[hidden_size, hidden_size],
                                        # initializer=tf.random_normal_initializer(stddev=hidden_stdv)
                                        )
        if name == 'hidden_state_in':
            w = tf.get_variable(
                name='w/hidden_state_in_' + str(i),
                shape=[hidden_size, hidden_size],
                # initializer=tf.random_normal_initializer(stddev=hidden_stdv)
            )

        if name == 'hidden_state_out':
            w = tf.get_variable(
                name='w/hidden_state_out_' + str(i),
                shape=[hidden_size, hidden_size],
                # initializer=tf.random_normal_initializer(stddev=hidden_stdv)
            )

        return w

    def biases(self, name, hidden_size, i):
        image_stdv = np.sqrt(1. / (2048))
        hidden_stdv = np.sqrt(1. / (hidden_size))
        # if name == 'hidden_state_out':
        #     b = tf.get_variable(name='b/hidden_state_out' + str(i), shape=[hidden_size],
        #                     initializer=tf.random_normal_initializer(stddev=hidden_stdv))
        #     # b = tf.get_variable(name='b/hidden_state_out', shape=[hidden_size],
        #     #                 initializer=tf.random_normal_initializer)
        # if name == 'hidden_state_in':
        #     b = tf.get_variable(name='b/hidden_state_in' + str(i), shape=[hidden_size],
        #                     initializer=tf.random_normal_initializer(stddev=hidden_stdv))
        #     # b = tf.get_variable(name='b/hidden_state_in', shape=[hidden_size],
        #     #                 initializer=tf.random_normal_initializer)
        # if name == 'out_image':
        #     # b = tf.get_variable(name='b/out_image_', shape=[2048],
        #     #                     initializer=tf.random_normal_initializer)
        #     b = tf.get_variable(name='b/out_image_' + str(i), shape=[2048],
        #                         initializer=tf.random_normal_initializer(stddev=image_stdv))
        if name == 'hidden_state':
            if i > 0:
                with tf.variable_scope("b", reuse=True):
                    b = tf.get_variable(name='hidden_state', shape=[hidden_size],
                                        initializer=tf.random_normal_initializer(stddev=hidden_stdv)
                                        )

            else:
                with tf.variable_scope("b"):
                    b = tf.get_variable(name='hidden_state', shape=[hidden_size],
                                        initializer=tf.random_normal_initializer(stddev=hidden_stdv)
                                        )
        if name == 'hidden_state_in':
            b = tf.get_variable(
                name='b/hidden_state_in_' + str(i),
                shape=(hidden_size, ),
                # initializer=tf.random_normal_initializer(stddev=hidden_stdv)
            )

        if name == 'hidden_state_out':
            b = tf.get_variable(
                name='b/hidden_state_out_' + str(i),
                shape=(hidden_size, ),
                # initializer=tf.random_normal_initializer(stddev=hidden_stdv)
            )

        return b

    def message_pass(self, x, hidden_size, batch_size, num_category, graph):

        w_hidden_state = self.weights('hidden_state_out', hidden_size, 0)
        b_hidden_state = self.biases('hidden_state_out', hidden_size, 0)
        x_all = tf.reshape(tf.matmul(
            tf.reshape(x[:, 0, :], [batch_size, hidden_size]),
            w_hidden_state) + b_hidden_state,
            [batch_size, hidden_size])

        for i in range(1, num_category):
            w_hidden_state = self.weights('hidden_state_out', hidden_size, i)
            b_hidden_state = self.biases('hidden_state_out', hidden_size, i)
            x_all_ = tf.reshape(tf.matmul(
                tf.reshape(x[:, i, :], [batch_size, hidden_size]),
                w_hidden_state) + b_hidden_state,
                [batch_size, hidden_size])
            x_all = tf.concat([x_all, x_all_], 1)
        x_all = tf.reshape(x_all, [batch_size, num_category, hidden_size])
        x_all = tf.transpose(x_all, (0, 2, 1))  # [batch_size, hidden_size, num_category]

        x_ = x_all[0]
        # graph_ = graph[0]
        x = tf.matmul(x_, graph)
        for i in range(1, batch_size):
            x_ = x_all[i]
            x_ = tf.matmul(x_, graph)
            x = tf.concat([x, x_], 0)
        x = tf.reshape(x, [batch_size, hidden_size, num_category])
        x = tf.transpose(x, (0, 2, 1))

        b_hidden_state = self.biases('hidden_state_in', hidden_size, 0)

        x_ = tf.reshape(
            tf.matmul(x[:, 0, :], self.weights('hidden_state_in', hidden_size, 0)) + b_hidden_state,
            [batch_size, hidden_size]
        )

        for j in range(1, num_category):
            b_hidden_state = self.biases('hidden_state_in', hidden_size, j)

            _x = tf.reshape(
                tf.matmul(x[:, j, :], self.weights('hidden_state_in', hidden_size, j)) + b_hidden_state,
                [batch_size, hidden_size]
            )
            x_ = tf.concat([x_, _x], 1)
        x = tf.reshape(x_, [batch_size, num_category, hidden_size])

        return x

    def GNN(self, feature_embedding_input, batch_size, hidden_size, n_steps, feature_num, graph):
        # feature_embedding_input (batch_size, feature_num, hidden_size)
        gru_cell = GRUCell(hidden_size)
        h0 = feature_embedding_input
        # print (h0)
        print [batch_size, feature_num, hidden_size]

        h0 = tf.reshape(h0, [batch_size, feature_num, hidden_size])
        # h0 = tf.nn.tanh(h0)
        state = h0
        # sum_graph = tf.reduce_sum(graph, reduction_indices=1)
        # enable_node = tf.cast(tf.cast(sum_graph, dtype=bool), dtype=tf.float32)

        with tf.variable_scope("gnn"):
            for step in range(n_steps):
                if step > 0: tf.get_variable_scope().reuse_variables()
                #state = state * mask_x
                x = self.message_pass(state, hidden_size, batch_size, feature_num, graph)

                (x_new, state_new) = gru_cell(x[0], state[0])
                # state_new = tf.transpose(state_new, (1,0))

                for i in range(1, batch_size):
                    (x_, state_) = gru_cell(x[i], state[i])  ##input of GRUCell must be 2 rank, not 3 rank
                    # state_ = tf.transpose(state_, (1,0))
                    state_new = tf.concat([state_new, state_], 0)
                #x = tf.reshape(x, [batch_size, num_category, hidden_size])
                state = tf.reshape(state_new, [batch_size, feature_num, hidden_size])  ##restore: 2 rank to 3 rank
            state = state + h0 

        return state, h0

    def attention_layer(self, state, w_attention, batch_size, hidden_size, num_category):
        # state size (batchsize, num_category, hiddensize)
        # with tf.variable_scope("attention", reuse=None):
        flat_state = tf.reshape(state, shape=(batch_size, hidden_size * num_category))
        return tf.sigmoid(tf.matmul(flat_state, w_attention))

    def  dense_layer(self, state, batch_size, hidden_size, num_category):

        flat_state = tf.reshape(state, shape=(batch_size, hidden_size * num_category))

        output = tf.contrib.layers.fully_connected(
            inputs=flat_state,
            num_outputs=400
        )
        output = tf.contrib.layers.fully_connected(
            inputs=output,
            num_outputs=1,
            activation_fn=None
        )
        return output

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.learning_rate, learning_rate))

def init_graph(config, loader):
    dimension = len(config.feature_num)
    # simple version
    graph = np.ones((dimension, dimension)) - np.eye(dimension)

    return graph.astype(np.float32)