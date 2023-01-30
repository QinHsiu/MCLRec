# MCLRec

This is our Pytorch implementation for the paper: "**Meta-optimized Contrastive Learning for Sequential Recommendation**".

## Environment  Requirement

* Pytorch>=1.7.0
* Python>=3.7 

## Usage

Please run the following command to install all the requirements:  

```python
pip install -r requirements.txt
```

## Datasets Prepare

Please use the `data_process.py` under `dataset/` to  get the input dataset by running the following command :

```python
python data_process.py
```

## Evaluate Model

We provide the trained models on Amazon_Beauty, Amazon_Sports_and_Outdoors, and Yelp datasets in `./log/Checkpoint/<Data_name>`folder. You can directly evaluate the trained models on test set by running:

```
python run_seq.py --dataset=<Data_name> --do_eval
```

On Amazon_Beauty:

```python
python run_seq.py --dataset=Amazon_Beauty --do_eval
```

```
 INFO  test result: {'recall@5': 0.0581, 'recall@10': 0.0871, 'recall@20': 0.1243, 'recall@50': 0.1852, 'mrr@5': 0.0278, 'mrr@10': 0.0316, 'mrr@20': 0.0341, 'mrr@50': 0.036, 'ndcg@5': 0.0352, 'ndcg@10': 0.0446, 'ndcg@20': 0.0539, 'ndcg@50': 0.066, 'precision@5': 0.0116, 'precision@10': 0.0087, 'precision@20': 0.0062, 'precision@50': 0.0037}

```

On Amazon_Sports_and_Outdoors:

```python
python run_seq.py --dataset=Amazon_Sports_and_Outdoors --do_eval
```

```
INFO  test result: {'recall@5': 0.0328, 'recall@10': 0.0501, 'recall@20': 0.0734, 'recall@50': 0.1215, 'mrr@5': 0.0163, 'mrr@10': 0.0186, 'mrr@20': 0.0202, 'mrr@50': 0.0218, 'ndcg@5': 0.0204, 'ndcg@10': 0.026, 'ndcg@20': 0.0319, 'ndcg@50': 0.0414, 'precision@5': 0.0066, 'precision@10': 0.005, 'precision@20': 0.0037, 'precision@50': 0.0024}
```

On Yelp:

```python
python run_seq.py --dataset=Yelp --do_eval
```

```
INFO  test result: {'recall@5': 0.0454, 'recall@10': 0.0647, 'recall@20': 0.0941, 'recall@50': 0.1557, 'mrr@5': 0.0292, 'mrr@10': 0.0317, 'mrr@20': 0.0337, 'mrr@50': 0.0356, 'ndcg@5': 0.0332, 'ndcg@10': 0.0394, 'ndcg@20': 0.0467, 'ndcg@50': 0.0589, 'precision@5': 0.0091, 'precision@10': 0.0065, 'precision@20': 0.0047, 'precision@50': 0.0031}
```

## Train Model

Please train the model using the Python script `run_seq.py`.

​	You can run the following command to train the model on Yelp datasets:

```
python run_seq.py --dataset=Yelp --epochs=100 --use_rl=1 --joint=0 train_batch_size=256 --lmd=0.03 --beta=0.1 --sim='dot'
```

