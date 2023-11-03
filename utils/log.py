
import copy
import logging


def my_logger(args):

    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    import os

    logger_path = f"{args.log_path}/fold{str(args.data_partition_idx)}/{args.full_experiment_name}.log"
    if not os.path.exists(f"{args.log_path}/fold{str(args.data_partition_idx)}"):
        os.makedirs(f"{args.log_path}/fold{str(args.data_partition_idx)}")
    if os.path.exists(logger_path) and not args.rewrite_log:
        assert False, "log file already exists!"


    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(logging.INFO)


    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)


    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, file_handler, console_handler


def get_logger():

    training_state = {
        "epoch": 0,
        "batch_idx": 0,
        "step": 0
    }

    results_test_epoch = {}
    results_valid_epoch = {}

    best_result = {
        "epoch": 0,
        "auc_pr": 0,
        "max_F1": 0,
        "bal_acc": 0,
        "kappa_score": 0,
        "positive_precision": 0,
        "positive_recall": 0,
        "positive_f1_score": 0,
        "negative_precision": 0,
        "negative_recall": 0,
        "negative_f1": 0,
    }

    best_results_test_auc_pr = copy.deepcopy(best_result)
    best_results_test_max_F1 = copy.deepcopy(best_result)
    best_results_test_bal_acc = copy.deepcopy(best_result)
    best_results_test_kappa_score = copy.deepcopy(best_result)
    final_results_test = copy.deepcopy(best_result)
    best_results_test_dict = {
        "auc_pr": best_results_test_auc_pr,
        "max_F1": best_results_test_max_F1,
        "bal_acc": best_results_test_bal_acc,
        "kappa_score": best_results_test_kappa_score,
        "final": final_results_test,
    }

    best_results_valid_auc_pr = copy.deepcopy(best_result)
    best_results_valid_max_F1 = copy.deepcopy(best_result)
    best_results_valid_bal_acc = copy.deepcopy(best_result)
    best_results_valid_kappa_score = copy.deepcopy(best_result)
    final_results_valid = copy.deepcopy(best_result)
    best_results_valid_dict = {
        "auc_pr": best_results_valid_auc_pr,
        "max_F1": best_results_valid_max_F1,
        "bal_acc": best_results_valid_bal_acc,
        "kappa_score": best_results_valid_kappa_score,
        "final": final_results_valid,
    }

    best_results_valid_auc_pr = copy.deepcopy(best_result)
    best_results_valid_max_F1 = copy.deepcopy(best_result)
    best_results_valid_bal_acc = copy.deepcopy(best_result)
    best_results_valid_kappa_score = copy.deepcopy(best_result)
    final_results_valid = copy.deepcopy(best_result)
    best_results_test_valid_dict = {
        "auc_pr": best_results_valid_auc_pr,
        "max_F1": best_results_valid_max_F1,
        "bal_acc": best_results_valid_bal_acc,
        "kappa_score": best_results_valid_kappa_score,
        "final": final_results_valid,
    }

    loss_epoch = {}
    GMM_state_epoch = {}

    return (
        training_state,
        results_test_epoch,
        results_valid_epoch,
        best_results_test_dict,
        best_results_valid_dict,
        # best_results_test_valid_dict,
        loss_epoch,
        GMM_state_epoch
    )
