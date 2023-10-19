cd ..

python run_seq.py --dataset=Amazon_Sports_and_Outdoors --hidden_dropout_prob=0.5 --attn_dropout_prob=0.5 --epochs=100 --use_rl=1 --joint=0 train_batch_size=256 --lmd=0.04 --beta=0.4 --sim='dot'
