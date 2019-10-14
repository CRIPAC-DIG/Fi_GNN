"""
find which config to use and select

"""


def Config(reader, flag="azenuz_small"):
    if flag == "azenuz_small":
        return configazenuz_small(reader)
    elif flag == "small_amazon":
        return smallconfig_amazon(reader)
    elif flag == "test_ptb":
        return testconfig_ptb(reader)
    elif flag == "small_amazontree":
        return smallconfig_amazontree(reader)
    elif flag == "small_amazontree1":
        return smallconfig_amazontree1(reader)
    elif flag == "small_amazontree2":
        return smallconfig_amazontree2(reader)
    elif flag == "small_amazontree3":
        return smallconfig_amazontree3(reader)
    else:
        raise ValueError("Invalid model: %s", flag)


class configazenuz_small(object):
    epoch_num = 10000
    train_batch_size = 1024  # 128
    train_step_size = 4 # 20
    valid_batch_size = 1024  # 20
    valid_step_size = 4
    hidden_size = 16  # 512

    lstm_forget_bias = 0.0
    # max_grad_norm = 0.25
    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 0.01  # 0.001  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 200
    sgd_opt = 'RMSProp'
    beta = 0.0001
    GNN_step = 3
    dropout_prob = 0
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
        # feature = loader.train_set[0]
        # print feature
        self.feature_flag = loader.feature_flag # [isinstance(each_feature, basestring) for each_feature in feature]
        self.feature_cat_num = [_num if _flag else None for _flag, _num in zip(*(self.feature_flag, self.feature_num))]
        self.feature_cat_num = filter(lambda x: x != None, self.feature_cat_num)
        self.feature_float_num = len(filter(lambda x: not x, self.feature_flag))

        print "feature_flag", self.feature_flag

        print "usernum", self.user_size
        print 'node_num', self.node_num
        print 'feature_num', self.feature_num
        print "maxstep_size %d" % self.maxstep_size
        print "gpu_id {}".format(self.gpu)
        print "learning_rate {}".format(self.learning_rate)


class smallconfig_amazon(object):
    epoch_num = 1000
    train_batch_size = 1  # 128
    train_step_size = 4 # 20
    valid_batch_size = 1  # 128
    valid_step_size = 4  # 20
    test_batch_size = 1  # 20
    test_step_size = 4

    def __init__(self, loader):
        vec = loader.itemdict.values()
        # print vec
        vec_r, vec_c = zip(*vec)
        self.vocab_size = (max(vec_r) + 2, max(vec_c) + 2)
        # self.vocab_size = loader.num_items  # 10000
        max_step = 0
        for line in loader.train_set:
            if max_step < len(line):
                max_step = len(line)
        self.maxstep_size = max_step + 1
        print "word-embedding %d" % self.word_embedding_dim


class smallconfig_amazontree(object):
    epoch_num = 1000
    train_batch_size = 100  # 128
    train_step_size = 4 # 20
    valid_batch_size = 100  # 128
    valid_step_size = 4  # 20
    test_batch_size = 100  # 20
    test_step_size = 4
    word_embedding_dim = 100  # 512
    lstm_layers = 1
    lstm_size = 100  # 512
    lstm_forget_bias = 0.0
    # max_grad_norm = 0.25
    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 1  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    dropout_prob = 0
    adagrad_eps = 1e-5
    gpu = 1

    def __init__(self, loader):
        vec = loader.itemdict.values()
        # vec_r, vec_c = zip(*vec)
        self.tree_size = len(zip(*vec)) - 1
        cat = [max(voc) + 2 for voc in zip(*vec)]
        self.vocab_size = tuple(cat)
        # self.vocab_size = loader.num_items  # 10000
        max_step = 0
        self.loader = loader
        for line in loader.train_set:
            if max_step < len(line):
                max_step = len(line)
        self.user_size = len(loader.train_set)
        self.maxstep_size = max_step + 1
        self.layer_embed = (0.2, 0.3, 0.3, 0.2)
        self.vocab_size_all = len(loader.itemdict)
        assert len(self.layer_embed) == self.tree_size
        print "usernum", self.user_size
        print 'itemnum_vocab_size_all', self.vocab_size_all
        print 'itemnum_vocab_size', self.vocab_size
        print "word-embedding %d" % self.word_embedding_dim


class smallconfig_amazontree1(smallconfig_amazontree):
    def __init__(self, loader):
        smallconfig_amazontree.__init__(self, loader)
        self.layer_embed = (0.1, 0.1, 0.3, 0.5)
        # self.word_embedding_dim = (self.word_embedding_dim / self.layer_embed[-1]) * sum(self.layer_embed)


class smallconfig_amazontree2(smallconfig_amazontree):
    def __init__(self, loader):
        smallconfig_amazontree.__init__(self, loader)
        self.layer_embed = (0, 0, 0, 1)
        # self.word_embedding_dim = (self.word_embedding_dim / self.layer_embed[-1]) * sum(self.layer_embed)


class smallconfig_amazontree3(smallconfig_amazontree):
    def __init__(self, loader):
        smallconfig_amazontree.__init__(self, loader)
        self.layer_embed = (0.6, 0.1, 0.1, 0.2)
        # self.word_embedding_dim = (self.word_embedding_dim / self.layer_embed[-1]) * sum(self.layer_embed)



class testconfig_ptb(object):
      """Tiny config, for testing."""
      init_scale = 0.1
      learning_rate = 1.0
      max_grad_norm = 1
      num_layers = 1
      num_steps = 2
      hidden_size = 2
      max_epoch = 1
      max_max_epoch = 1
      keep_prob = 1.0
      lr_decay = 0.5
      batch_size = 20

      def __init__(self, reader):
          self.vocab_size = len(reader.vocab.words)  # 10000
