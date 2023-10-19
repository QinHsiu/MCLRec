cd ..

python run_seq.py --dataset=Amazon_Beauty --epochs=100 --hidden_dropout_prob=0.5 --attn_dropout_prob=0.5 --use_rl=1 --joint=0 train_batch_size=256 --lmd=0. --beta=0.05 --sim='dot'
