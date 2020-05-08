import os
import pickle
from datetime import *
from collections import deque
import platform
import numpy as np
import tensorflow as tf
from util.load_data_movielens import Loader, INFO_LOG
from Config import Config
from util.LearningRateUpdater import LearningRateUpdater
from model.model_fignn import FiGNN


def run(session, config, model, loader, dropout, verbose=False):
    total_cost = 0.
    total_auc = 0.
    num_ = 0.
    mean_p = 0.
    mean_n = 0.
    c_p = 0.
    c_n = 0.
    for batch in loader.generate_batch_data(batchsize=model.batch_size, mode=model.mode):

        batch_id, batch_num, feature_batch, label_batch = batch

        feed = {
            model.input_x: feature_batch,
            model.input_y: label_batch,
            model.dropout: dropout
        }

        # print model.input_x, feature_batch
        # print model.input_y, label_batch

        out = [model.cost, model.optimizer, model.auc_result,
               model.auc_opt, model.predict]

        output = session.run(out, feed)
        # print output
        cost, _, auc, _, s_pos = output

        if model.mode == "Train":
            auc = 0.
        total_cost += cost
        total_auc = auc
        # total_mayauc += may_auc
        # print list(label_batch)
        # print s_pos

        num_ += 1.
        if verbose and batch_id % int(batch_num / 5.) == 1 and model.mode == "Valid":
            INFO_LOG("{}/{}, cost: {}, auc: {}".format(
                batch_id, batch_num, total_cost / num_,
                auc  # total_auc / num_
            )
            )
        if model.mode == "Valid":
            for idx in range(model.batch_size):
                label = int(label_batch[idx][0])
                prediction = s_pos[idx][0]
                # print label, prediction
                # print label
                # print label == 1, label == 0
                if label == 1:
                    mean_p += prediction
                    c_p += 1.
                elif label == 0:
                    mean_n += prediction
                    c_n += 1.
                    # print c_p, c_n
    if model.mode == "Valid":
        print ("mean_p : {},  mean_n : {}".format(mean_p / c_p, mean_n / c_n))

    return total_cost / num_, total_auc


def main(_):

    loader = Loader(flag="movielens")
    config = Config(loader, flag="movielens")
    config.gpu = 2
    if platform.system() == 'Linux':
        gpuid = config.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
        device = '/gpu:' + str(gpuid)
    else:
        device = '/cpu:0'

    lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_epoch)

    i = 0
    graph = tf.Graph()
    with graph.as_default():
        trainm = FiGNN(config, device, loader, "Train")
        testm = FiGNN(config, device, loader, "Valid")

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=session_config) as session:
        session.run(tf.global_variables_initializer())
        dropout = [1.0 - config.dropout_prob]
        no_dropout = [1.0]
        best_auc = 0.
        best_auc_queue = 0.
        best_epoch_auc = 0.
        maxlen = 20
        auc_queue = deque(maxlen=maxlen)
        auc = 0.
        for epoch in range(config.epoch_num):
            trainm.update_lr(session, lr_updater.get_lr())
            cost, auc = run(session, config, trainm, loader, dropout, verbose=True)
            INFO_LOG("Epoch %d Train AUC %.4f" % (epoch + 1, auc))
            INFO_LOG("Epoch %d Train costs %.4f" %
                     (epoch, cost))
            session.run(tf.local_variables_initializer())

            cost, auc = run(session, config, testm, loader, no_dropout, verbose=True)
            INFO_LOG("Epoch %d Valid AUC %.4f" % (epoch, auc))
            INFO_LOG("Epoch %d Valid cost %.4f" % (epoch, cost))

            auc_queue.append(auc)

            lr_updater.update(auc, epoch)
            if best_auc < auc:
                best_auc = auc
                best_epoch_auc = epoch


            mean_auc = np.mean(auc_queue)
            if  best_auc_queue < mean_auc:
                best_auc_queue = mean_auc



            INFO_LOG("*** best and mean AUC now are %.4f, %.4f, in %d epoch" % (best_auc, best_auc_queue, best_epoch_auc))

            # only 1 split, no need to change
            # if epoch % 1000 == 0 and epoch != 0:
            #     loader.change_data_list(loader.increase_data_idx())


if __name__ == '__main__':
    tf.app.run()
