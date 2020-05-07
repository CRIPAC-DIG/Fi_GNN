"""
find which config to use and select

"""


def Config(loader, flag="avazu"):
    if flag == "avazu":
        return config_avazu(loader)
    elif flag == "criteo":
        return config_criteo(loader)
    elif flag == "movielens":
        return config_movielens(loader)
    else:
        raise ValueError("Invalid dataset config: %s", flag)

class config_avazu(object):
    epoch_num = 10000
    train_batch_size = 1024  # 128
    train_step_size = 4 # 20
    valid_batch_size = 1024  # 20
    valid_step_size = 4
    emb_size = 32
    hidden_size1 = 32 # hidden_size of dnn
    hidden_size2 = 32 # the final output size


    lstm_forget_bias = 0.0
    # max_grad_norm = 0.25
    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 0.001  # 0.001  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 200
    sgd_opt = 'RMSProp'
    beta = 1e-6
    GNN_step = 3
    dropout_prob = 0.2  #keep_prob = 1-dropout_prob
    adagrad_eps = 1e-5
    gpu = 1

    def __init__(self, loader):
        self.node_num = len(loader.train_set[0]) # the num of features
        self.feature_num = loader.feature_num
        assert self.node_num == len(self.feature_num)

        self.user_size = len(loader.train_set)
        max_step = 0
        for line in loader.train_set:
            if max_step < len(line):
                max_step = len(line)
        self.maxstep_size = max_step + 1

        # split the categorical and float value
        feature = loader.train_set[0]
        # print feature
        if hasattr(loader, 'feature_flag'):
            self.feature_flag = loader.feature_flag # [isinstance(each_feature, basestring) for each_feature in feature]
        else:
            self.feature_flag = [isinstance(each_feature, basestring) for each_feature in feature]
        self.feature_cat_num = [_num if _flag else None for _flag, _num in zip(*(self.feature_flag, self.feature_num))]
        self.feature_cat_num = filter(lambda x: x != None, self.feature_cat_num)
        self.feature_float_num = len(filter(lambda x: not x, self.feature_flag))

        print ("feature_flag", self.feature_flag)

        print ("usernum", self.user_size)
        print ('node_num', self.node_num)
        print ('feature_num', self.feature_num)
        print ("maxstep_size %d" % self.maxstep_size)
        print ("gpu_id {}".format(self.gpu))
        print ("learning_rate {}".format(self.learning_rate))


class config_criteo(object):
    epoch_num = 10000
    train_batch_size = 1024  # 128
    train_step_size = 4 # 20
    valid_batch_size = 1024  # 20
    valid_step_size = 4
    emb_size = 32
    hidden_size1 = 32  # hidden_size of dnn
    hidden_size2 = 32  # the final output size

    lstm_forget_bias = 0.0
    # max_grad_norm = 0.25
    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 0.001  # 0.001  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 200
    sgd_opt = 'RMSProp'
    beta = 1e-6
    GNN_step = 3
    dropout_prob = 0.2
    adagrad_eps = 1e-5
    gpu = 1

    def __init__(self, loader):
        self.node_num = len(loader.train_set[0]) # the num of features
        self.feature_num = loader.feature_num
        assert self.node_num == len(self.feature_num)

        self.user_size = len(loader.train_set)
        max_step = 0
        for line in loader.train_set:
            if max_step < len(line):
                max_step = len(line)
        self.maxstep_size = max_step + 1

        # split the categorical and float value
        feature = loader.train_set[0]
        # print feature
        if hasattr(loader, 'feature_flag'):
            self.feature_flag = loader.feature_flag # [isinstance(each_feature, basestring) for each_feature in feature]
        else:
            self.feature_flag = [isinstance(each_feature, basestring) for each_feature in feature]
        self.feature_cat_num = [_num if _flag else None for _flag, _num in zip(*(self.feature_flag, self.feature_num))]
        self.feature_cat_num = filter(lambda x: x != None, self.feature_cat_num)
        self.feature_float_num = len(filter(lambda x: not x, self.feature_flag))

        print ("feature_flag", self.feature_flag)

        print ("usernum", self.user_size)
        print ('node_num', self.node_num)
        print ('feature_num', self.feature_num)
        print ("maxstep_size %d" % self.maxstep_size)
        print ("gpu_id {}".format(self.gpu))
        print ("learning_rate {}".format(self.learning_rate))


class config_movielens(object):
    epoch_num = 10000
    train_batch_size = 1024  # 128
    train_step_size = 4 # 20
    valid_batch_size = 1024  # 20
    valid_step_size = 4
    emb_size = 16
    hidden_size1 = 32  # hidden_size of dnn
    hidden_size2 = 32  # the final output size

    lstm_forget_bias = 0.0
    # max_grad_norm = 0.25
    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 0.001  # 0.001  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 200
    sgd_opt = 'RMSProp'
    beta = 1e-6
    GNN_step = 3
    dropout_prob = 0.2
    adagrad_eps = 1e-5
    gpu = 1

    def __init__(self, loader):
        self.node_num = len(loader.train_set[0]) # the num of features
        self.feature_num = loader.feature_num
        assert self.node_num == len(self.feature_num)

        self.user_size = len(loader.train_set)
        max_step = 0
        for line in loader.train_set:
            if max_step < len(line):
                max_step = len(line)
        self.maxstep_size = max_step + 1

        # split the categorical and float value
        feature = loader.train_set[0]
        # print feature
        if hasattr(loader, 'feature_flag'):
            self.feature_flag = loader.feature_flag # [isinstance(each_feature, basestring) for each_feature in feature]
        else:
            self.feature_flag = [isinstance(each_feature, basestring) for each_feature in feature]
        self.feature_cat_num = [_num if _flag else None for _flag, _num in zip(*(self.feature_flag, self.feature_num))]
        self.feature_cat_num = filter(lambda x: x != None, self.feature_cat_num)
        self.feature_float_num = len(filter(lambda x: not x, self.feature_flag))

        print ("feature_flag", self.feature_flag)

        print ("usernum", self.user_size)
        print ('node_num', self.node_num)
        print ('feature_num', self.feature_num)
        print ("maxstep_size %d" % self.maxstep_size)
        print ("gpu_id {}".format(self.gpu))
        print ("learning_rate {}".format(self.learning_rate))
