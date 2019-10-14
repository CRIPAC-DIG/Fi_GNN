import numpy as np
import json
import pickle
import random
import time
import platform
import os.path as path
import os


attribte_list = [u'site_id', u'C20', u'C19', u'site_domain', u'device_type',
                     u'id', u'C17', u'click', u'device_ip', u'C14', u'C16', u'C15',
                     u'device_conn_type', u'C1', u'app_category', u'site_category',
                     u'app_domain', u'C21', u'banner_pos', u'app_id', u'device_id',
                     u'hour', u'device_model', u'C18']


def INFO_LOG(info):
    print "[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info)


def init_data(fpath, data_index):

    filpath_name, file_name = path.split(fpath)
    print filpath_name
    attr2id = dict(zip(*(attribte_list, range(len(attribte_list)))))

    with open(fpath, 'r') as f:
        original_data = json.load(f)
    # sorted by hour
    print len(original_data[0][0])

    for idx in range(len(original_data)):
        original_data[idx] = sorted(original_data[idx], key=lambda x: x[attr2id['hour']])

    # split label
    for idx in range(len(original_data)):
        original_data[idx] = map(lambda x: [x[attr2id['click']], x], original_data[idx])

    # del feature
    for idx in range(len(original_data)):
        for jdx in range(len(original_data[idx])):
            del original_data[idx][jdx][1][attr2id['click']]
            # del original_data[idx][jdx][1][attr2id['device_ip']]
            del original_data[idx][jdx][1][attr2id['id']]
            # del original_data[idx][jdx][1][attr2id['device_id']]

    # split train and test
    train = []
    test = []
    for idx in range(len(original_data)):
        train.append(original_data[idx][0:-1])
        test.append(original_data[idx][-1])

    with open(filpath_name + '/' + 'train_' + str(data_index) + '.json', 'w') as f:
        f.write(json.dumps(train))
    with open(filpath_name + '/' + 'valid_' + str(data_index) + '.json', 'w') as f:
        f.write(json.dumps(test))


class Loader(object):
    def __init__(self, flag):
        if platform.system() == 'Linux':
            self.path_init_file = path_init_file = '/home/cuizeyu/pythonfile/CTR_GNN/criteo/cut_to_smaller_size/'
            self.path_file = path_file = '/home/cuizeyu/pythonfile/CTR_GNN/criteo/cut_to_smaller_size/'
            # path_init_file = '/home/hufenyu/Documents/pythonfile/dataset/'
            # path_file = '/home/hufenyu/Documents/pythonfile/lightrnn_czy/data/'

        else:
            self.path_init_file = path_init_file = '/home/cuizeyu/pythonfile/CTR_GNN/criteo/'
            self.path_file = path_file = '/home/cuizeyu/pythonfile/CTR_GNN/criteo/'

        self.data_split_num = 100

        # init loader train_0 valid_0
        self.data_idx = 50
        self.change_data_list(self.data_idx)

    def increase_data_idx(self):
        self.data_idx = (self.data_idx + 1) % self.data_split_num
        if  self.data_idx !=0 and  self.data_idx < 10:
            self.data_idx = 10
        return self.data_idx

    def change_data_list(self, idx):
        trainfile_name = 'train_' + str(idx) + '.json'
        validfile_name = 'valid.json'

        print('==================Change the loader data list===========================')
        print
        print('----------------{}------------------------------'.format(trainfile_name))
        print('----------------{}------------------------------'.format(validfile_name))

        with open(self.path_file + trainfile_name, 'r') as f:
            self.train_data = train_data = json.load(f)

        with open(self.path_file + validfile_name, 'r') as f:
            self.valid_data = valid_data = json.load(f)

        # split feature and label
        self.train_set = []
        self.train_label = []

        for a_sequence in train_data:
            label_temp, feature_temp = a_sequence  # [label, list], right
            self.train_set.append(feature_temp)  # list in list, right
            self.train_label.append(label_temp)
        self.valid_label, self.valid_set = zip(*valid_data)
        self.valid_set = list(self.valid_set)
        self.valid_label = list(self.valid_label)
        print "test"
        print 'self.train_set[0]', self.train_set[0], self.train_set[1] # correct list
        print self.train_label[0], self.train_label[1]  # correct value
        print "_test"
        # calculate feature nums

        # feature = self.train_set[0]
        # self.feature_flag = [isinstance(each_feature, basestring) for each_feature in feature]
        self.feature_flag = [False] * 13 + [True] * 26

        def _map_instance(x):
            for idx in range(39):
                if not self.feature_flag[idx]:
                    if x[idx] != u'<empty>':
                        x[idx] = float(x[idx])
            return x

        self.train_set = map(_map_instance, self.train_set)
        self.valid_set = map(_map_instance, self.valid_set)

        each_feature_num, each_feature_id = self.load_feature_id(self.path_file+'each_feature_id.json')
        self.feature_num = each_feature_num
        self.each_feature_id = each_feature_id

        print "len_train_set", len(self.train_set), "len_train_label", len(self.train_label)
        print "len_valid_set", len(self.valid_set), "len_valid_label", len(self.valid_label)

        print "re_id_all_feature"
        print 'self.train_set[0]', self.train_set[0] # correct
        # input()
        self.train_set, self.train_label = self.re_id(self.train_set, self.train_label)
        self.valid_set, self.valid_label = self.re_id(self.valid_set, self.valid_label)
        print "finlish_re_id"

        print "len_train_set", len(self.train_set), "len_train_label", len(self.train_label)
        print "len_valid_set", len(self.valid_set), "len_valid_label", len(self.valid_label)

        # print self.train_set
        print "each_feature_num"
        print each_feature_num

        # divide the 1 0 label
        temp_zip = zip(*(self.train_set, self.train_label))
        # print temp_zip

        temp_zip_1 = filter(lambda x: str(x[1]) == '1', temp_zip)
        temp_zip_0 = filter(lambda x: str(x[1]) == '0', temp_zip)
        # print temp_zip_1
        self.train_set_1, _ = zip(*temp_zip_1)
        self.train_set_0, _ = zip(*temp_zip_0)
        # print self.train_set_1

        print "label_1_num: {}, label_0_num: {}".format(
            len(self.train_set_1), len(self.train_set_0)
        )

        # # sequence data generate
        # print "user_train_sequence", len(self.train_data)

        # # each_feature_id

        # print "user_train_sequence_after_filter", len(self.seq_all_data)

        print('==================================================================')

    def load_feature_id(self, path_):
        each_feature_id = []
        with open(path_, 'r') as f:
            each_feature_id = json.load(f)
        # each_feature_id = each_feature_id.values()


        feature_num = [len(temp) for temp in each_feature_id]
        return feature_num, each_feature_id

    def generate_batch_data(self, batchsize, mode):
        if mode == "Train":
            dataset = self.train_set
            datalabel = self.train_label
        elif mode == "Valid":
            dataset = self.valid_set
            datalabel = self.valid_label

        INFO_LOG("mode: %s" % mode)
        batch_id = 0
        batch_num = len(dataset) / batchsize
        t = -1
        feature_batch = []
        label_batch = []
        for idx, feature in enumerate(dataset):
            t += 1
            if t > 0 and t % batchsize == 0:
                yield (batch_id, batch_num,
                       np.asarray(feature_batch).reshape((batchsize, len(self.feature_num))),
                       np.asarray(label_batch).reshape((batchsize, 1))
                       )

                label_batch = []
                feature_batch = []
                batch_id += 1
                t = -1

            else:
                if mode == "Valid":
                    feature_batch.append(feature)
                    label_batch.append(datalabel[idx])
                else:
                    if random.random() > 1./ 2.:
                        feature = random.choice(self.train_set_1)
                        feature_batch.append(feature)
                        # label_batch.append(datalabel[idx])
                        label_batch.append(1)
                    else:
                        feature = random.choice(self.train_set_0)
                        feature_batch.append(feature)
                        # label_batch.append(datalabel[idx])
                        label_batch.append(0)

    def generate_sequence_data(self, batchsize, mode):

        # if mode == "Train":
        #     dataset = self.train_set
        #     datalabel = self.train_label
        # elif mode == "Valid":
        #     dataset = self.valid_set
        #     datalabel = self.valid_label
        batchsize = 1

        INFO_LOG("mode: %s" % mode)
        batch_id = 0
        batch_num = len(self.seq_all_data) / batchsize
        t = -1
        feature_batch = []
        label_batch = []

        for idx, whole_seq in enumerate(self.seq_all_data):
            if mode == "Train":
                input_seq = whole_seq[0:-1]
            elif mode == "Valid":
                input_seq = whole_seq

            label_batch, feature_batch = zip(*input_seq)
            yield (batch_id, batch_num,
                   np.asarray(feature_batch).reshape((-1, len(self.feature_num))),
                   np.asarray(label_batch).reshape((-1, 1))
                   )

    def re_id(self, dataset, label):
        idx = 0
        while idx < len(dataset):
            feature = dataset[idx]
            # print self.each_feature_id
            # print len(feature)
            for jdx, each_feature in enumerate(feature):
                # print  jdx, self.each_feature_id[jdx]
                if self.feature_flag[jdx]:
                    if self.each_feature_id[jdx].has_key(each_feature):
                        # print self.each_feature_id[jdx][each_feature]
                        dataset[idx][jdx] = self.each_feature_id[jdx][each_feature]
                    else:
                        del dataset[idx]
                        del label[idx]
                        idx -= 1
                        break
            idx += 1

        return dataset, label

    def seq_rid(self, train, valid):
        seq_all_data = []
        assert len(train) == len(valid)
        for idx, (train_seq, valid_seq) in enumerate(zip(*(train, valid))):
            whole_seq = train_seq + [valid_seq]
            # print "WHOLE_SEQ", whole_seq
            idx = 0
            while idx < len(whole_seq):
                # print whole_seq[idx]
                label, feature = whole_seq[idx]
                for jdx, each_feature in enumerate(feature):
                    if self.each_feature_id[jdx].has_key(each_feature):
                        whole_seq[idx][1][jdx] = self.each_feature_id[jdx][each_feature]
                    else:
                        del whole_seq[idx]
                        idx -= 1
                        break
                idx += 1
            if len(whole_seq) > 3:
                seq_all_data.append(whole_seq)

        return seq_all_data

    def split_feature(self, feature_batch, config):
        # example_feature = feature_batch[0]
        # new_flag = [isinstance(each_feature, basestring) for each_feature in example_feature]
        # feature_flag = config.feature_flag
        # for idx in range(len(new_flag)):
        #     assert new_flag[idx] == feature_flag[idx]
        new_flag = config.feature_flag
        index_true_flag = filter(lambda x: new_flag[x], range(len(new_flag)))
        index_false_flag = filter(lambda x: not new_flag[x], range(len(new_flag)))

        temp_feature_batch = zip(*feature_batch)
        cate_feature = zip(*map(lambda x: temp_feature_batch[x], index_true_flag))
        float_feature = zip(*map(lambda x: temp_feature_batch[x], index_false_flag))
        # cate_feature = [feature_batch[index_true_flag] for one_feature in feature_batch]
        # float_feature = [feature_batch[index_false_flag] for one_feature in feature_batch]
        # cate_feature = filter(feature_batch, )
        return cate_feature, float_feature






if __name__ == "__main__":
    # temp_graph = np.asarray([[[0,1,1,0], [1,0,1,0], [1,1,0,0], [0,0,0,0]]])
    # reuniform_graph(temp_graph)
    # load_train_data(2, 16)
    loader = Loader(flag='haha')
