
import os
import nni
import json
import copy
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    precision_recall_curve,
    cohen_kappa_score,
    balanced_accuracy_score,
    classification_report,
    auc
)

from torch.utils.tensorboard import SummaryWriter

# 我的package
from utils.log import my_logger, get_logger
from datasets.dataloader_DeepDDs import DeepDDs_dataset, DeepDDs_dataloader
from models.DeepDDs import GCNNet

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # self.args.full_experiment_name = f"{self.args.experiment_name}_fold{str(self.args.data_partition_idx)}_lr{str(self.args.learning_rate)[:8]}_epochs{str(self.args.epochs)}_warmup{str(self.args.warmup_noisylabel)}"
        # self.args.full_experiment_name = f"{self.args.experiment_name}_epochs{str(self.args.epochs)}_warmup{str(self.args.warmup_noisylabel)}"
        self.args.full_experiment_name = f"{self.args.experiment_name}_epochs{str(self.args.epochs)}"
        # if self.args.noisylabel: self.args.full_experiment_name = self.args.full_experiment_name + "_noisylabel"
        # self.args.fold_experiment_name = f"{self.args.experiment_name}_fold{str(self.args.data_partition_idx)}"

        # self.CE = nn.CrossEntropyLoss(reduction="none").to(self.args.device)
        # self.CEloss = torch.nn.CrossEntropyLoss().to(self.args.device)
        
        self.CE = nn.functional.binary_cross_entropy_with_logits
        self.CEloss = nn.functional.binary_cross_entropy_with_logits

        self.dataset = DeepDDs_dataset(args, type='warmup')
        self.data_loader = DeepDDs_dataloader(self.args, self.dataset)

        self.drug1_data_train = self.data_loader.drug1_data_train
        self.drug2_data_train = self.data_loader.drug2_data_train
        self.drug1_loader_train = self.data_loader.drug1_loader_train
        self.drug2_loader_train = self.data_loader.drug2_loader_train

        self.drug1_loader_train_evaluate = self.data_loader.drug1_loader_train_evaluate
        self.drug2_loader_train_evaluate = self.data_loader.drug2_loader_train_evaluate

        if self.args.evaluate_valid:
            self.drug1_data_valid = self.data_loader.drug1_data_valid
            self.drug2_data_valid = self.data_loader.drug2_data_valid
            self.drug1_loader_valid = self.data_loader.drug1_loader_valid
            self.drug2_loader_valid = self.data_loader.drug2_loader_valid

        self.drug1_data_test = self.data_loader.drug1_data_test
        self.drug2_data_test = self.data_loader.drug2_data_test
        self.drug1_loader_test = self.data_loader.drug1_loader_test
        self.drug2_loader_test = self.data_loader.drug2_loader_test

        self.args.data_state = {}
        for key, item in self.dataset.data_state.items():
            if "number" in key:
                self.args.data_state[key] = item

        self.model = GCNNet(args).to(self.args.device)
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

    def train_epoch(self, dataloader_1, dataloader_2):

        total_loss = 0
        current_epoch = self.training_state["epoch"]

        self.model.train()

        for batch_idx, data in enumerate(zip(dataloader_1, dataloader_2)):

            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(self.args.device)
            data2 = data2.to(self.args.device)

            self.training_state["batch_idx"] = batch_idx

            target = data[0].y.view(-1, 1).long().to(self.args.device)
            target = target.squeeze(1)

            self.optimizer.zero_grad()
            output = self.model(data1, data2)
            loss = self.CEloss(output, target.float())

            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()


            self.training_state["step"] += 1
            self.training_state["batch_idx"] = batch_idx

        total_loss /= len(dataloader_1)

        self.logger.info(f"loss: {total_loss}")
        self.writer.add_scalars(
            f"loss/loss",
            {
                f"loss": total_loss
            },
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
            
            self.train_epoch(self.drug1_loader_train, self.drug2_loader_train)
            if self.args.evaluate_valid: self.predicting("valid", self.drug1_loader_valid, self.drug2_loader_valid)
            self.predicting("test", self.drug1_loader_test, self.drug2_loader_test)
            self.save_results()
            self.save_checkpoint()
        # self.wandb_writer.finish()
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.console_handler)

    def predicting(self, test_type, drug1_loader, drug2_loader):
        self.model.eval()


        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_prelabels = torch.Tensor()
        test_results = {}

        with torch.no_grad():
            for data in zip(drug1_loader, drug2_loader):
                data1 = data[0]
                data2 = data[1]
                data1 = data1.to(self.args.device)
                data2 = data2.to(self.args.device)
                output = self.model(data1, data2)

                # ys = F.softmax(output, 1).to('cpu').data.numpy()
                ys = F.sigmoid(output).to('cpu').data.numpy()
                # ys = output.to('cpu').data.numpy()
                
                # predicted_labels = list(map(lambda x: np.argmax(x), ys))
                # predicted_scores = list(map(lambda x: x[1], ys))
                
                predicted_labels = list(ys > 0.5)
                predicted_scores = list(ys)
                
                total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
                total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
                total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

        total_labels = total_labels.numpy().flatten()
        total_preds = total_preds.numpy().flatten()
        total_prelabels = total_prelabels.numpy().flatten()
        # self.writer.add_pr_curve(f'PR_Curve_{str(self.training_state["epoch"])}', np.array(Y_true), np.array(Y_pred_probs), self.training_state["epoch"])

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
