import time


def INFO_LOG(info):
    print ("[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info))


class LearningRateUpdater(object):
    def __init__(self, init_lr, decay_rate, decay_when):
        self._init_lr = init_lr
        self._decay_rate = decay_rate
        self._decay_when = decay_when
        self._current_lr = init_lr
        self._last_auc = -1

    def get_lr(self):
        return self._current_lr

    def update(self, cur_auc, epoch):
        # if self._last_auc > 0 and cur_auc - self._last_auc < self._decay_when:
        #     current_lr = self._current_lr * self._decay_rate
        #     INFO_LOG("learning rate: {} ==> {}".format(self._current_lr, current_lr))
        #     self._current_lr = current_lr
        # self._last_auc = cur_auc
        if epoch == self._decay_when:
            self._current_lr = 0.001
            INFO_LOG("learning rate: {} ==> {}".format(self._current_lr, self._last_auc))

        self._last_auc = cur_auc

