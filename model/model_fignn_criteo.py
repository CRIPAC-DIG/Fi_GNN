import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers
from functools import partial


"fignn for criteo"

class FiGNN(object):
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

        self.emb_size = emb_size = config.emb_size
        self.hidden_size1 = hidden_size1 = config.hidden_size1
        self.hidden_size2 = hidden_size2 = config.hidden_size2
        self.GNN_step = GNN_step = config.GNN_step
        self.node_num = node_num = config.node_num

        opt = config.sgd_opt
        beta = config.beta
        batch_size = self.batch_size

        hidden_stdv = np.sqrt(1. / (emb_size))
        # embedding initial
        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):
            feature_embedding_list = []
            for idx, each_feature_num in enumerate(config.feature_cat_num):
                feature_embedding_list.append(

                    tf.get_variable(name='feature_embedding_' + str(idx), shape=[each_feature_num, emb_size],
                                    initializer=tf.random_normal_initializer(hidden_stdv))
                )

            float_feature_embedding = tf.get_variable(
                name='float_feature_embedding', shape=[config.feature_float_num, emb_size],
                initializer=tf.random_normal_initializer(hidden_stdv)
            )

        # #------------feed-----------------##
        print(len(config.feature_cat_num))
        print(config.feature_float_num)
        self.input_x = input_x = tf.placeholder(tf.int32, [batch_size, len(config.feature_cat_num)])
        self.input_x_float = input_x_float = tf.placeholder(tf.float32, [batch_size, config.feature_float_num])
        self.input_y = input_y = tf.placeholder(tf.int32, [batch_size, 1])
        self.dropout = dropout = tf.placeholder(tf.float32, [1])

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
        print(len(feature_embedding_input))
        for idx in range(len(input_x_float_unstack)):
            # print tf.nn.embedding_lookup(float_feature_embedding, idx).shape
            # print input_x_float_unstack[idx].shape
            float_feature_embedding_idx = tf.nn.embedding_lookup(float_feature_embedding, idx)
            stack_batch_float_feature_embedding_idx = tf.stack(batch_size * [float_feature_embedding_idx])
            # print stack_batch_float_feature_embedding_idx.shape
            # print input_x_float_unstack[idx]
            feature_embedding_input.append(
                stack_batch_float_feature_embedding_idx * tf.reshape(input_x_float_unstack[idx], (batch_size, 1))
            )
        print(len(feature_embedding_input))

        self.feature_input = tf.transpose(tf.stack(feature_embedding_input, axis=0), (1, 0, 2))

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):

            graph = self.init_graph(config, self.feature_input)

            final_state, initial_state = self.GNN(
                self.feature_input, batch_size, emb_size, GNN_step, len(config.feature_num), graph
            )  # output: [batch_size, config.feature_num, hiddensize]

            self.predict = predict = self.prediction_layer(final_state)

        # -------------evaluation--------------
        self.auc_result, self.auc_opt = tf.metrics.auc(
            labels=self.input_y,
            predictions=predict
        )
        # self.score = score
        # -------------cost ---------------
        score_mean = tf.losses.log_loss(
            labels=self.input_y,
            predictions=predict
        )
        self.cost = cost = score_mean

        # ---------------optimizer---------------#
        self.no_opt = tf.no_op()
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False)

        if mode == 'Train':
            self.auc_opt = tf.no_op()
            self.auc_result = tf.no_op()
            if opt == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
            if opt == 'Momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(cost)
            if opt == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
            if opt == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cost)

        else:
            self.optimizer = tf.no_op()


    def message_pass(self, x, hidden_size, batch_size, node_num, graph):

        states_1 = []
        for i in range(node_num):
            state = tf.reshape(
                tf.layers.dense(tf.reshape(x[:, i, :], [batch_size, hidden_size]), hidden_size, activation=None, use_bias=None,
                                name="out_"+str(i)), [batch_size, hidden_size])
            states_1.append(state)
        states_1 = tf.stack(states_1, axis=2)  # [batch_size, hidden_size, node_num]

        states_2 = []
        for i in range(batch_size):
            state = tf.matmul(states_1[i], graph[i])
            states_2.append(state)
        states_2 = tf.stack(states_2, axis=0)
        states_2 = tf.transpose(states_2, (0, 2, 1)) # [batch_size, node_num, hidden_size]

        states_3 = []
        for i in range(node_num):
            state = tf.reshape(
                tf.layers.dense(tf.reshape(states_2[:, i, :], [batch_size, hidden_size]), hidden_size, activation=None,
                                use_bias=None,
                                name="in_" + str(i)), [batch_size, hidden_size])
            states_3.append(state)
        states_3 = tf.stack(states_3, axis=2)  # [batch_size, hidden_size, node_num]
        states_3 = tf.transpose(states_3, (0, 2, 1))  # [batch_size, node_num, hidden_size]

        return states_3


    def GNN(self, feature_embedding_input, batch_size, hidden_size, n_steps, node_num, graph):
        # feature_embedding_input ( batch_size, feature_num, hidden_size)
        gru_cell = GRUCell(hidden_size)
        h0 = feature_embedding_input
        # print (h0)
        h0 = tf.reshape(h0, [batch_size, node_num, hidden_size])
        # h0 = tf.nn.tanh(h0)
        state = h0

        # states = []
        # states.append(state)

        with tf.variable_scope("gnn"):
            for step in range(n_steps):
                if step > 0: tf.get_variable_scope().reuse_variables()
                x = self.message_pass(state, hidden_size, batch_size, node_num, graph)
                # states = []
                # for i in range(batch_size):
                #     (x_, state_) = gru_cell(x[i], state[i])
                #     states.append(state_)
                # state_new = tf.stack(states, axis=0)
                (x_new, state_new) = gru_cell(tf.reshape(x, [-1, hidden_size]), tf.reshape(state, [-1, hidden_size]))
                state_new = tf.reshape(state_new, [self.batch_size, self.node_num, self.emb_size])
                #Residual
                state = h0 + state_new

                # states.append(state)

        return state, h0

    def readout(self, state, i):
        with tf.variable_scope("readout"):
            if i > 0: tf.get_variable_scope().reuse_variables()
            g1 = tf.layers.dense(state, self.hidden_size1, activation=tf.nn.leaky_relu, use_bias=True, name="weights_g1")
            g1 = tf.nn.dropout(g1, self.dropout[0])
            # g2 = tf.layers.dense(g1, self.hidden_size1, activation=tf.nn.leaky_relu, use_bias=True, name="weights_g2")
            # g2 = tf.nn.dropout(g2, self.dropout[0])
            # g3 = tf.layers.dense(g2, self.hidden_size2, activation=tf.nn.leaky_relu, use_bias=True, name="weights_g3")
            # g3 = tf.nn.dropout(g3, self.dropout[0])
            # g4 = tf.layers.dense(g3, self.hidden_size2, activation=tf.nn.leaky_relu, use_bias=True, name="weights_g4")
            # g4 = tf.nn.dropout(g4, self.dropout[0])
            g5 = tf.layers.dense(g1, self.hidden_size2, activation=tf.nn.tanh, use_bias=True, name="weights_g5")
            g5 = tf.nn.dropout(g5, self.dropout[0])
            output = tf.reduce_max(g5, axis=1) + tf.reduce_sum(g5, axis=1)
            #output = tf.reduce_sum(g, axis=1)

        return output

    def prediction_layer(self, state):
        a = tf.layers.dense(tf.reshape(state, [-1, self.emb_size]), 1, activation=tf.nn.sigmoid, use_bias=None, name="atten")
        a = tf.reshape(a, [self.batch_size, self.node_num])
        s = tf.layers.dense(tf.reshape(state, [-1, self.emb_size]), 1, activation=partial(tf.nn.leaky_relu, alpha=0.01), use_bias=None, name="score")
        s = tf.reshape(s, [self.batch_size, self.node_num])
        predict = tf.sigmoid(tf.reshape(tf.reduce_sum(s * a, axis=1), [self.batch_size, 1]))
        return predict


    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.learning_rate, learning_rate))

    def init_graph(self, config, x):

        dimension = len(config.feature_num)
        # simple version
        graph = np.ones((dimension, dimension)) - np.eye(dimension)
        # generate a batch
        graph = np.tile(graph.reshape((1, dimension, dimension)), (self.batch_size, 1, 1))

        return graph.astype(np.float32)




