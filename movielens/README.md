### Fi-GNN code for movielens dataset

This is the tensorflow implementation of **Fi-GNN** on movielens dataset which has a multi-value field (genre).

You should first run the data processing code for movielens in `data/preprocess.py`

#### Preprocessing

```
cd data
python preprocess.py
```

#### Train and Test

```
CUDA_VISIBLE_DEVICES=1 python -m code.train \
                        --model_type FiGNN \
                       --data_path data  \
                       --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save --has_residual \
                       --save_path ./models/movie/fignn_64x64x64/ \
                       --field_size 7  --run_times 1 \
                       --dropout_keep_prob "[0.6, 0.9]" \
                       --epoch 50 --batch_size 1024 \

```

