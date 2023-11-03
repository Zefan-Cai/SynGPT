
import os
import nni
import json
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_recall_curve,
    cohen_kappa_score,
    balanced_accuracy_score,
    classification_report,
    auc
)
from utils.utils import set_seed

# 日志文件
import wandb
wandb.login()
from torch.utils.tensorboard import SummaryWriter

# 我的package
from utils.log import my_logger, get_logger
from models.DeepSynergy import DeepSynergy
from datasets.dataloader_DeepSynergy import DeepSynergy_dataset, DeepSynergy_dataloader

class Trainer(object):
    def __init__(self, args):

        set_seed(args)
        self.args = args
        self.args.full_experiment_name = f"{self.args.experiment_name}_epochs{str(self.args.epochs)}"

        self.CE = nn.functional.binary_cross_entropy_with_logits
        self.CEloss = nn.functional.binary_cross_entropy_with_logits


        self.dataset = DeepSynergy_dataset(self.args, type='warmup')
        self.data_loader = DeepSynergy_dataloader(self.args, self.dataset)

        self.data_loader_train = self.data_loader.drug_loader_train
        
        if self.args.evaluate_valid:
            self.data_loader_valid = self.data_loader.drug_loader_valid
            
        self.data_loader_test  = self.data_loader.drug_loader_test


        self.model = DeepSynergy().to(self.args.device)
        self.params = [{'params': self.model.parameters()}]
        self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
        
        (
            self.training_state,
            self.results_test_epoch,
            self.results_valid_epoch,
            self.best_results_test_dict,
            self.best_results_valid_dict,
            self.loss_epoch,
            self.GMM_state_epoch,
        ) = get_logger()
        
        self.logger, self.file_handler, self.console_handler = my_logger(self.args)

        tensorboard_dir = f"{self.args.tensorboard_path}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
        if not os.path.exists(
            f"{self.args.tensorboard_path}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.tensorboard_path}/fold{str(self.args.data_partition_idx)}"
            )
        if os.path.exists(tensorboard_dir) and not self.args.rewrite_log:
            assert False, "tensorboard directory already exists!"
            
        self.writer = SummaryWriter(tensorboard_dir)

        for key, value in vars(args).items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("\n")


    def train_epoch(self, data_loader):

        total_loss = 0
        current_epoch = self.training_state["epoch"]

        self.model.train()

        for batch_idx, data_ori in enumerate(data_loader):
            data = data_ori[:, :-1].to(self.args.device)
            drug1, drug2, cell = data[:, 0:256 + 346 + 200], data[:, (256 + 346 + 200):(256 + 346 + 200)*2], data[:, (256 + 346 + 200) * 2:]
            assert drug1.size(1) == drug2.size(1)
            # pdb.set_trace()
            assert cell.size(1) == 37261
            # data_swap = torch.cat([drug2, drug1, cell], dim=1)
            # data = torch.cat([data, data_swap], dim=0)
            y = data_ori[:, -1].view(-1, 1).float().to(self.args.device)
            # y = y.squeeze(1)
            # y = torch.cat([y, y])

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.CEloss(output.squeeze(), y.squeeze())
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()


            self.training_state["step"] += 1

        total_loss /= len(data_loader)

        self.logger.info(f"loss: {total_loss}")
        self.writer.add_scalars(
            f"loss/loss",
            {f"loss": total_loss},
            self.training_state["epoch"],
        )
        # self.wandb_writer.log({f"loss": total_loss}, step=self.training_state["epoch"])
        self.loss_epoch[self.training_state["epoch"]] = total_loss



    def train(self):

        self.logger.info("Train")
        # self.wandb_writer = wandb.init(project=self.args.fold_experiment_name, name=self.args.full_experiment_name, config=self.args)
        # self.wandb_writer.watch(self.model)
        
        for epoch in range(0, self.args.epochs):
            
            self.training_state["epoch"] = epoch
            self.logger.info(f"epoch: {epoch}")
            
            self.train_epoch(self.data_loader_train)
            
            if self.args.evaluate_valid: self.predicting("valid", self.data_loader_valid)
            
            self.predicting("test", self.data_loader_test)
            
            self.save_results()
            
        # self.wandb_writer.finish()
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.console_handler)
        
    def predicting(self, test_type, data_loader):
        self.model.eval()


        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_prelabels = torch.Tensor()
        test_results = {}

        with torch.no_grad():
            for data_ori in data_loader:
                data = data_ori[:, :-1].to(self.args.device)
                y = data_ori[:, -1].view(-1, 1).long().to(self.args.device)

                output = self.model(data)
                
                
                
                
                # ys = F.softmax(output, 1).to('cpu').data.numpy()
                # predicted_labels = list(map(lambda x: np.argmax(x), ys))
                # predicted_scores = list(map(lambda x: x[1], ys))
                
                ys = F.sigmoid(output).to("cpu").data.numpy()
                predicted_labels = list(ys > 0.5)
                predicted_scores = list(ys)
                
                
                
                total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
                total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
                total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)





        precision, recall, thresholds = precision_recall_curve(
            total_labels, total_preds
        )
        test_results["auc_pr"] = auc(recall, precision)
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores[recall == 0] = 0
        f1_scores[precision == 0] = 0
        test_results["max_F1"] = np.max(f1_scores).item()
        test_results["bal_acc"] = balanced_accuracy_score(
            total_labels, total_prelabels
        ).item()
        test_results["kappa_score"] = cohen_kappa_score(
            total_labels, total_prelabels
        ).item()
        classifi_report = classification_report(
            total_labels, total_prelabels, target_names=[0, 1], output_dict=True
        )
        test_results["positive_precision"] = classifi_report[1]["precision"]
        test_results["positive_recall"] = classifi_report[1]["recall"]
        test_results["positive_f1_score"] = classifi_report[1]["f1-score"]
        test_results["negative_precision"] = classifi_report[0]["precision"]
        test_results["negative_recall"] = classifi_report[0]["recall"]
        test_results["negative_f1"] = classifi_report[0]["f1-score"]

        current_epoch = self.training_state["epoch"]
        if test_type == "test":
            self.results_test_epoch[f"epoch_{str(current_epoch)}"] = test_results
            results_epoch = self.results_test_epoch
            best_results_dict = self.best_results_test_dict
            # self.wandb_writer.log(test_results, step=self.training_state["epoch"])
            for key in test_results.keys():
                self.writer.add_scalars(
                    f"{test_type}/{key}",
                    {"NoisyLabel": test_results[key]},
                    self.training_state["epoch"],
                )
        elif test_type == "valid":
            self.results_valid_epoch[f"epoch_{str(current_epoch)}"] = test_results
            results_epoch = self.results_valid_epoch
            best_results_dict = self.best_results_valid_dict

        current_epoch = self.training_state["epoch"]
        for key, value in results_epoch[f"epoch_{str(current_epoch)}"].items():
            self.logger.info(f"{test_type} {key}: {value}")
        self.logger.info("\n")

        for key in ["auc_pr", "max_F1", "bal_acc", "kappa_score"]:
            if (
                results_epoch[f"epoch_{str(current_epoch)}"][key]
                > best_results_dict[key][key]
            ):
                best_results_dict[key] = results_epoch[
                    f"epoch_{str(current_epoch)}"
                ]
                best_results_dict[key]["epochs"] = current_epoch
                self.save_checkpoint(metrics=key)




    def save_results(self):
        directory = f"{self.args.result_path}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(
            f"{self.args.result_path}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.result_path}/fold{str(self.args.data_partition_idx)}"
            )

        if os.path.exists(directory) and not self.args.rewrite_log:
            assert False, "result directory already exists!"

        final_epoch = self.training_state["epoch"]
        self.best_results_test_dict["final"] = self.results_test_epoch[
            f"epoch_{str(final_epoch)}"
        ]

        # results_epoch
        with open(directory + "/results_test_epoch.json", "w") as fp:
            json.dump(self.results_test_epoch, fp)
        # best_results_epoch
        with open(directory + "/best_results_test_dict.json", "w") as fp:
            json.dump(self.best_results_test_dict, fp)

        with open(directory + "/GMM_state_epoch.json", "w") as fp:
            json.dump(self.GMM_state_epoch, fp)

        with open(directory + "/loss_epoch.json", "w") as fp:
            json.dump(self.loss_epoch, fp)

        if self.args.evaluate_valid:
            self.best_results_valid_dict["final"] = self.results_valid_epoch[
                f"epoch_{str(final_epoch)}"
            ]
            with open(directory + "/results_valid_epoch.json", "w") as fp:
                json.dump(self.results_valid_epoch, fp)
            with open(directory + "/best_results_valid_dict.json", "w") as fp:
                json.dump(self.best_results_valid_dict, fp)
            # with open(directory + "/best_results_test_valid_dict.json", "w") as fp:
            #     json.dump(self.best_results_test_valid_dict, fp)

        nni.report_final_result(
            self.results_test_epoch[f"epoch_{str(final_epoch)}"]["auc_pr"]
        )
        

            

    def save_checkpoint(self, metrics="", model_index=0, type=''):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        model = self.models[model_index]
        optimizer = self.optimizers[model_index]
        current_epoch = self.training_state["epoch"]
        state = {
            "epoch": current_epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        # if save_best:

        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
        if type != '': directory += f"_{type}"

        if not os.path.exists(
            f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
            )
        if not os.path.exists(directory):
            os.makedirs(directory)

        if metrics != "":
            best_path = f"{directory}/checkpoint_{str(model_index)}_best_{metrics}.pth"
            torch.save(state, best_path)
            self.logger.info(f"Saving current best {metrics}: {best_path}")
        else:
            filename = f"{directory}/checkpoint_{str(model_index)}_epoch{str(current_epoch)}.pth"
            torch.save(state, filename)
            self.logger.info(f"Saving epoch {str(current_epoch)} checkpoint: {filename}")
        
        
        
        

    def save_GMM_state(self):
        current_epoch = self.training_state["epoch"]
        self.GMM_state_epoch[f"epoch_{str(current_epoch)}"] = self.dataset.data_state_evaluate
        for key, value in self.GMM_state_epoch[f"epoch_{str(current_epoch)}"].items():
            self.logger.info(f"{key}: {value}")

        for key in self.dataset.data_state_evaluate.keys():
            self.writer.add_scalars(
                f"GMM_state/{key}",
                {
                    "train": self.dataset.data_state_evaluate[key]
                },
                self.training_state["epoch"],
            )
        # self.wandb_writer.log(self.dataset.data_state_evaluate, step=self.training_state["epoch"])





    def save_checkpoint(self, metrics=""):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        model = self.model
        optimizer = self.optimizer
        current_epoch = self.training_state["epoch"]
        
        state = {
            "epoch": current_epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        # if save_best:

        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"

        if not os.path.exists(
            f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
            )
        if not os.path.exists(directory):
            os.makedirs(directory)

        if metrics != "":
            best_path = f"{directory}/checkpoint_best_{metrics}.pth"
            torch.save(state, best_path)
            self.logger.info(f"Saving current best {metrics}: model_best.pth ...")
        else:
            filename = f"{directory}/checkpoint_epoch{str(current_epoch)}.pth"
            torch.save(state, filename)
            self.logger.info(f"Saving epoch {str(current_epoch)} checkpoint: {filename} ...")
