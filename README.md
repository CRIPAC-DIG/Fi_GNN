
# UPDATE 2022.01.08
The code has been moved to this repo [GraphCTR](https://github.com/CRIPAC-DIG/GraphCTR), which includes the graph-based CTR prediction models and some other representative baselines.


# FiGNN for CTR prediction

The code and data for our paper in CIKM2019: Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction,
[arxiv](https://arxiv.org/abs/1910.05552).



<div align=center>
  <img src="https://github.com/CRIPAC-DIG/Fi_GNN/blob/74ca80e9ca459c4641d7fe10a70fccb081ef7daa/figures/model.png" width = 50% height = 50% />
</div>
The input sparse multi-field feature vector is first mapped into sparse one-hot embedding vectors and then embedded to dense field embedding vectors via the embedding layer and the multi-head self-attention layer. These field embedding vectors are then represented as a feature graph, where each node corresponds to a feature field and different feature fields can interact through edges. The task of modeling interaction can be thus converted to modeling node interactions on the feature graph. Therefore, the feature graph is feed into our proposed Fi-GNN to model node interactions. An attention scoring layer is applied on the output of Fi-GNN to estimate the click- through rate.


Next, we introduce how to run FiGNN on four benchmark data sets.

## Requirements: 
* **Tensorflow 1.5.0**
* Python 3.6
* CUDA 9.0+ (For GPU)

## Usage
Our code is based on Weiping Song and Chence Shi's [AutoInt](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec).
### Input Format
The required input data is in the following format:
* train_x: matrix with shape *(num_sample, num_field)*. train_x[s][t] is the feature value of feature field t of sample s in the dataset. The default value for categorical feature is 1.
* train_i: matrix with shape *(num_sample, num_field)*. train_i[s][t] is the feature index of feature field t of sample s in the dataset. The maximal value of train_i is the feature size.
* train_y: label of each sample in the dataset.

If you want to know how to preprocess the data, please refer to `data/Dataprocess/Criteo/preprocess.py`

### Example
There are four public real-world datasets (Avazu, Criteo, KDD12, MovieLens-1M) that you can choose. You can run the code on MovieLens-1M dataset directly in `/movielens`. The other three datasets are super huge, and they can not be fit into the memory as a whole. Therefore, we split the whole dataset into 10 parts and we use the first file as test set and the second file as valid set. We provide the codes for preprocessing these three datasets in `data/Dataprocess`. If you want to reuse these codes, you should first run `preprocess.py` to generate `train_x.txt, train_i.txt, train_y.txt` as described in `Input Format`. Then you should run `data/Dataprocesss/Kfold_split/StratifiedKfold.py` to split the whole dataset into ten folds. Finally you can run `scale.py` to scale the numerical value(optional).

To help test the correctness of the code and familarize yourself with the code, we upload the first `10000` samples of `Criteo` dataset in `train_examples.txt`. And we provide the scripts for preprocessing and training.(Please refer to `	data/sample_preprocess.sh` and `run_criteo.sh`, you may need to modify the path in `config.py` and `run_criteo.sh`). 

After you run the `data/sample_preprocess.sh`, you should get a folder named `Criteo` which contains `part*, feature_size.npy, fold_index.npy, train_*.txt`. `feature_size.npy` contains the number of total features which will be used to initialize the model. `train_*.txt` is the whole dataset. If you use other small dataset, say `MovieLens-1M`, you only need to modify the function `_run_` in `autoint/train.py`.

Here's how to run the preprocessing.

```
cd data
mkdir Criteo
python ./Dataprocess/Criteo/preprocess.py
python ./Dataprocess/Kfold_split/stratifiedKfold.py
python ./Dataprocess/Criteo/scale.py
```

Besides our proposed model FiGNN, you can also choose AutoInt model. You should specify the model type (FiGNN or AutoInt) when running the training.
 
Here's how to run the training.

```
CUDA_VISIBLE_DEVICES=0 python -m code.train \
                        --model_type FiGNN \
                        --data_path data --data Criteo \
                        --blocks 3 --heads 2 --block_shape "[64,64,64]" \
                        --is_save --has_residual \
                        --save_path ./models/Criteo/fignn_64x64x64/ \
                        --field_size 39  --run_times 1 \
                        --epoch 3 --batch_size 1024 \
```

You should see the output like this:

```
...
train logs
...
start testing!...
restored from ./models/Criteo/b3h2_64x64x64/1/
test-result = 0.8088, test-logloss = 0.4430
test_auc [0.8088305055534442]
test_log_loss [0.44297631300399626]
avg_auc 0.8088305055534442
avg_log_loss 0.44297631300399626
```

## Citation
If you find FiGNN useful for your research, please consider citing the following paper:
```
@inproceedings{li2019fi,
  title={Fi-gnn: Modeling feature interactions via graph neural networks for ctr prediction},
  author={Li, Zekun and Cui, Zeyu and Wu, Shu and Zhang, Xiaoyu and Wang, Liang},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={539--548},
  year={2019}
}
```


## Contact information
You can contact Zekun Li (`lizekunlee@gmail.com`), if there are questions related to the code.


## Acknowledgement
This implementation is based on Weiping Song and Chence Shi's [AutoInt](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec). Thanks for their sharing and contribution.
