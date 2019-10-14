import os
import pickle
from datetime import *
import random
import platform
import numpy as np
import tensorflow as tf
from util.load_data_score_graph import Loader, INFO_LOG

from Config import Config
from util.load_data_score_graph import Loader
from util.model_saver import CTR_GNN_saver, CTR_GNN_loader
from util.LearningRateUpdater import LearningRateUpdater
from model.model_res_A_h0_forvisual import CTR_ggnn


def run(session, config, model, loader, verbose=False):
    total_cost = 0.
    total_auc = 0.
    total_mayauc = 0.
    num_ = 0.
    mean_p = 0.
    mean_n = 0.
    c_p = 0.
    c_n = 0.
    len_feature = len(config.feature_num)
    avg_node_att = np.zeros((len_feature,))
    avg_node_score = np.zeros((len_feature,))
    avg_edge_attention = np.zeros((len_feature, len_feature))

    tdx = 0

    for batch in loader.generate_batch_data(batchsize=model.batch_size, mode=model.mode):


        batch_id, batch_num, feature_batch, label_batch = batch

        feed = {
            model.input_x: feature_batch,
            model.input_y: label_batch
        }

        # print model.input_x, feature_batch
        # print model.input_y, label_batch

        out = [model.cost, model.optimizer, model.auc_result,
               model.auc_opt, model.predict, model.node_attention, model.node_score, model.edge_attention]

        output = session.run(out, feed)
        # print output
        cost, _, auc, _, s_pos, node_attention, node_score, edge_attention = output
        # avg_node_att
        avg_node_att += np.mean(node_attention, 0)
        # print "nodeatt", np.mean(node_attention, 0).shape
        avg_node_score += np.mean(node_score, 0)
        # print "nodescore", np.mean(node_score, 0).shape
        avg_edge_attention += np.mean(edge_attention, 0)
        # print "edge_attention", np.mean(edge_attention, 0).shape

        for idx in range(len(s_pos)):
            if int(label_batch[idx]) == 1 and s_pos[idx] > 0.85 and random.uniform(0, 1) > 0.9:
                np.save('./case/case_node_attention_' + str(tdx) + '_' + str(s_pos[idx])[2:6], node_attention[idx])
                tdx += 1


        if model.mode == "Train":
            auc = 0.
        total_cost += cost
        total_auc = auc
        # total_mayauc += may_auc
        # print list(label_batch)
        # print s_pos

        num_ += 1.
    avg_node_att = avg_node_att / num_
    avg_node_score = avg_node_score / num_
    avg_edge_attention = avg_edge_attention / num_

    np.save('avg_node_att', avg_node_att)
    np.save('avg_node_score', avg_node_score)
    np.save('avg_edge_attention', avg_edge_attention)
    print "*****************************************"

    return total_cost / num_, total_auc


def main(_):

    loader = Loader(flag="azenuz_small")
    config = Config(loader, flag="azenuz_small")
    config.gpu = 1
    if platform.system() == 'Linux':
        gpuid = config.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
        device = '/gpu:' + str(gpuid)
    else:
        device = '/cpu:0'

    config.gpu = 0 # for load the model
    lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_epoch)

    i = 0
    graph = tf.Graph()
    with graph.as_default():
        trainm = CTR_ggnn(config, device, loader, "Train")
        testm = CTR_ggnn(config, device, loader, "Valid")

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=session_config) as session:
        # session.run(tf.global_variables_initializer())
        CTR_GNN_loader(session, config)
        best_auc = 0.
        best_logloss = 1.
        best_epoch_auc = 0.
        best_epoch_logloss = 0.
        auc = 0.
        for epoch in range(1):
            # trainm.update_lr(session, lr_updater.get_lr())
            # cost, auc = run(session, config, trainm, loader, verbose=True)
            # INFO_LOG("Epoch %d Train AUC %.3f" % (epoch + 1, auc))
            # INFO_LOG("Epoch %d Train costs %.3f" %
            #          (epoch, cost))
            session.run(tf.local_variables_initializer())

            cost, auc = run(session, config, testm, loader, verbose=True)
            INFO_LOG("Epoch %d Valid AUC %.3f" % (epoch, auc))
            INFO_LOG("Epoch %d Valid cost %.3f" % (epoch, cost))
            # #

            lr_updater.update(auc, epoch)
            if best_auc < auc:
                best_auc = auc
                best_epoch_auc = epoch
                # CTR_GNN_saver(session, config, best_auc, best_epoch_auc)

            if best_logloss > cost:
                best_logloss = cost
                best_epoch_logloss = epoch
                # CTR_GNN_saver(session, config, best_epoch_logloss, best_epoch_logloss)


            INFO_LOG("*** best AUC now is %.3f in %d epoch" % (best_auc, best_epoch_auc))
            INFO_LOG("*** best logloss now is %.3f in %d epoch" % (best_logloss, best_epoch_logloss))

            if epoch % 300 == 0 and epoch != 0:
                loader.change_data_list(loader.increase_data_idx())


if __name__ == '__main__':
    tf.app.run()
