cd ..

python run_seq.py --dataset=ml-1m --epochs=100 --hidden_dropout_prob=0.1 --attn_dropout_prob=0.1 --use_rl=1  --joint=0 train_batch_size=256 --lmd=0. --beta=0.01 --sim='dot'
