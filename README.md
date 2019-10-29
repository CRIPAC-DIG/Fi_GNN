# Fi_GNNs

The code and dataset for our paper in the CIKM2019:Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction [[arXiv version]](https://arxiv.org/pdf/1910.05552.pdf)

## Paper data and code

This is the code for the CIKM-2019 Paper: [Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction](https://arxiv.org/pdf/1910.05552.pdf). We have implemented our methods in **Tensorflow**.

### criteos

### avazu



## Usage
### the data preprocess is written in the `./data/README.md` 


Then you can run the file `NGNN/main_score.py` to train the model.

You can change parameters according to the usage in `NGNN/Config.py`:

```bash

parameters arguments in `NGNN/Config.py`:

    epoch_num           the max epoch number
    train_batch_size    training batch size
    valid_batch_size    validation batch size
    hidden_size         hidden size of the NGNN
    lstm_forget_bias    forget bias in NGNN update
    max_grad_norm       the gradient clip during train
    init_scale          the scale of initialize parameter 0.05
    learning_rate       learning rate  0.01  # 0.001  # 0.2
    decay               the decay of 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 200
    sgd_opt             train strategy can choose: 'RMSProp', 'Adam', 'Momentum', 'RMSProp', 'Adadelta'
    beta                the weight of regulartion
    GNN_step            the number of step of GNN
    dropout_prob        the dropout probability of our model
    adagrad_eps         eps
    gpu = 0             the gpu id
                        
                        
                        
```

## Requirements

- Python 2.7
- Tensorflow 1.5.0

## Citation

Please cite our paper if you use the code:

```
@article{li2019fi,
  title={Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction},
  author={Li, Zekun and Cui, Zeyu and Wu, Shu and Zhang, Xiaoyu and Wang, Liang},
  journal={arXiv preprint arXiv:1910.05552},
  year={2019}
}
```

