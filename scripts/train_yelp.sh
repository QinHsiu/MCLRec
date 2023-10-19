cd ..

python run_seq.py --dataset=Yelp --epochs=100 --hidden_dropout_prob=0.5 --attn_dropout_prob=0.5 --use_rl=1 --joint=0 train_batch_size=256 --lmd=0.03 --beta=0.1 --sim='dot'
