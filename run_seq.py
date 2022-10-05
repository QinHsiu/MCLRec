import argparse

from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MCLRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='gpu id')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument("--do_eval", action="store_true")
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list,do_eval=args.do_eval)
