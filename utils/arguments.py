
import torch
import argparse

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", default="gmm_sepe_simple", type=str, help="")
    parser.add_argument("--start_fold", default=3, type=int, help="")
    parser.add_argument("--end_fold", default=4, type=int, help="")

    parser.add_argument("--result_path", default="results/", help="Path to log.")
    parser.add_argument("--log_path", default="log/", help="Path to log.")
    parser.add_argument("--tensorboard_path", default="tensorboard_log/", help="Path to log.")
    parser.add_argument("--checkpoint_dir", default="tensorboard_log/", help="Path to log.")
    parser.add_argument("--output_dir", default="tensorboard_log/", help="Path to log.")


    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--epochs", default=50, type=int, help="")
    parser.add_argument("--warmup", default=10, type=int, help="warmup schedule")
    parser.add_argument("--model", default="GraphSynergy", type=str, help="DeepDDs, GraphSynergy") # 选择使用什么模型

    parser.add_argument("--learning_rate", default=0.0005, type=float, help="")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="")
    parser.add_argument("--amsgrad", default=False, type=bool, help="")

    parser.add_argument("--gamma", default=0.1, type=float, help="")
    parser.add_argument("--step_size", default=20, type=int, help="")

    parser.add_argument("--evaluate_valid", default=False, type=bool, help="")

    parser.add_argument("--noisylabel", default=False, type=bool, help="")
    parser.add_argument("--warmup_noisylabel", default=10, type=int, help="warmup for noisy label")
    parser.add_argument('--p_threshold_positive', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--p_threshold_negative', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument("--seperate_clean_noise_noisylabel", default=False, type=bool, help="")
    parser.add_argument("--warmup_dataset", type=str, default="warmup", help="")
    parser.add_argument("--co_refinement", default=False, type=bool, help="")
    parser.add_argument("--penalty_loss", default=False, type=bool, help="")
    parser.add_argument("--prior", default=0.06, type=float, help="")
    parser.add_argument("--negative_GMM", default=True, type=bool, help="")
    parser.add_argument("--noisylabel_only_noisy", default=False, type=bool, help="")
    
    parser.add_argument("--rewrite_log", default=False, type=bool, help="")
    parser.add_argument("--best_metrics", default="auc_pr", type=str, help="")

    parser.add_argument("--data_dir", type=str, default="/home/czf/UW/data_journel/GraphSynergy/Data", help="")
    parser.add_argument("--warmup_data_dir", type=str, default="/home/czf/UW/data_journel/GraphSynergy/Data", help="")
    parser.add_argument("--noisylabel_data_dir", type=str, default="", help="")
    parser.add_argument("--noisylabel_warmup_data_dir", type=str, default="/home/czf/UW/data_journel/GraphSynergy/Data_Train_TestRandom", help="")
    parser.add_argument("--aux_data_dir", type=str, default="/home/czf/UW/data_journel/GraphSynergy/Data", help="")
    
    parser.add_argument("--noisy_data_ratio", default=2, type=float, help="")
    parser.add_argument("--part_shuffle", default=False, type=bool, help="")
    parser.add_argument("--warmup_sample_number", default=1, type=int, help="")
    parser.add_argument("--ReloadCkpt", default="first", type=str, help="first, every, none")
    parser.add_argument("--save_data", default=False, type=bool, help="")
    parser.add_argument("--Sharpen", default=False, type=bool, help="")
    parser.add_argument("--Sharpen_Alpha", default=2, type=float, help="")
    parser.add_argument("--SoftLabel", default="None", type=str, help="None, CoRefine, Sharpen")


    # GRAPHSYNERGY
    ## data_loader
    parser.add_argument("--score", type=str, default="synergy 0", help="")
    parser.add_argument("--n_hop", type=int, default=2, help="")
    parser.add_argument("--n_memory", type=int, default=128, help="")
    
    # PRODeepSyn
    parser.add_argument("--hidden_size", type=int, default=2048, help="")

    ## model
    parser.add_argument("--emb_dim", type=int, default=64, help="")
    parser.add_argument("--l1_decay", type=float, default=0.000006, help="")
    parser.add_argument("--therapy_method", type=str, default="transformation_matrix", help="")

    args = parser.parse_args()

    device = torch.device(f"cuda:{str(args.gpuid)}")
    args.device = device


    return args



