
import os
import nni
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.metrics import (
    precision_recall_curve,
    cohen_kappa_score,
    balanced_accuracy_score,
    classification_report,
    auc
)
from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter

from utils.log import my_logger, get_logger
from datasets.dataloader_AuDNNsynergy import (
    AuDNNsynergy_dataset,
    AuDNNsynergy_dataloader,
)
from models.AuDNNsynergy import DNN
from utils.utils import set_seed


class Trainer(object):
    def __init__(self, args):
        self.args = args
        set_seed(args)
        self.args.full_experiment_name = f"{self.args.experiment_name}_epochs{str(self.args.epochs)}"
        self.args.fold_experiment_name = self.args.full_experiment_name

        tensorboard_dir = f"{self.args.tensorboard_path}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"

        # self.CE = nn.BCEWithLogitsLoss()
        # self.CEloss = nn.BCEWithLogitsLoss()


        self.CE = nn.functional.binary_cross_entropy_with_logits
        self.CEloss = nn.functional.binary_cross_entropy_with_logits

        # warmup
        self.warmup_dataset = AuDNNsynergy_dataset(self.args, type="warmup")
        self.warmup_dataloader = AuDNNsynergy_dataloader(self.args, self.warmup_dataset)
        
        self.warmup_dataset_train = self.warmup_dataset.data_train
        self.warmup_dataloader_train = self.warmup_dataloader.loader_train
        
        if self.args.evaluate_valid:
            self.dataset_valid = self.warmup_dataset.data_valid
            self.dataloader_valid = self.warmup_dataloader.loader_valid
            
        self.dataset_test = self.warmup_dataset.data_test
        self.dataloader_test = self.warmup_dataloader.loader_test
        

        


        self.model = DNN(self.warmup_dataset_train.cell_feat_len() + 2 * self.warmup_dataset_train.drug_feat_len(), self.args.hidden_size).to(self.args.device)
        self.params = [{"params": self.model.parameters()}]
        self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)


        (
            self.training_state,
            self.results_test_epoch,
            self.results_valid_epoch,
            self.best_results_test_dict,
            self.best_results_valid_dict,
            self.loss_epoch,
            self.GMM_state_epoch
        ) = get_logger()

        self.logger, self.file_handler, self.console_handler = my_logger(self.args)

        self.writer = SummaryWriter(tensorboard_dir)
        if not os.path.exists(
            f"{self.args.tensorboard_path}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.tensorboard_path}/fold{str(self.args.data_partition_idx)}"
            )
        if os.path.exists(tensorboard_dir) and not self.args.rewrite_log:
            assert False, "tensorboard directory already exists!"

        for key, value in vars(args).items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("\n")

    def train_epoch(self, data_loader):
        total_loss = 0
        current_epoch = self.training_state["epoch"]

        self.model.train()
        for batch_idx, batch in enumerate(data_loader):
            
            drug1_feats, drug2_feats, cell_feats, y_true = batch
            drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.to(self.args.device), drug2_feats.to(self.args.device), cell_feats.to(self.args.device), y_true.to(self.args.device)
            y_pred = self.model(drug1_feats, drug2_feats, cell_feats)
            loss = self.CEloss(y_pred, y_true)

            self.training_state["batch_idx"] = batch_idx

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

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
        # self.wandb_writer = wandb.init(
        #     project=self.args.fold_experiment_name,
        #     name=self.args.full_experiment_name,
        #     config=self.args,
        # )
        # self.wandb_writer.watch(self.model)

        for epoch in range(0, self.args.epochs):
            self.training_state["epoch"] = epoch
            self.logger.info(f"epoch: {epoch}")

            self.train_epoch(self.warmup_dataloader_train)
            # self.save_checkpoint()

            if self.args.evaluate_valid:
                self.predicting("valid", self.dataloader_valid)
            self.predicting("test", self.dataloader_test)

            self.save_results()

        # self.wandb_writer.finish()
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.console_handler)

    def predicting(self, test_type, data_loader):
        self.model.eval()

        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_prelabels = torch.Tensor()
        # total_data = torch.Tensor()
        total_outputs = torch.Tensor()
        test_results = {}
        
        current_epoch = self.training_state["epoch"]

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):

                drug1_feats, drug2_feats, cell_feats, y_true = batch
                drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.to(self.args.device), drug2_feats.to(self.args.device), cell_feats.to(self.args.device), y_true.to(self.args.device)
                
                y_pred = self.model(drug1_feats, drug2_feats, cell_feats)
                ys = F.sigmoid(y_pred).to("cpu").data.numpy()
                
                # y_pred_1 = self.model(drug1_feats, drug2_feats, cell_feats)
                # y_pred_2 = self.model(drug2_feats, drug1_feats, cell_feats)
                # y_pred = (y_pred_1 + y_pred_2) / 2
                # ys = F.sigmoid(y_pred).to("cpu").data.numpy()
                
                predicted_labels = list(ys > 0.5)
                predicted_scores = list(ys)
                total_preds = torch.cat(
                    (total_preds, torch.Tensor(predicted_scores)), 0
                )
                total_prelabels = torch.cat(
                    (total_prelabels, torch.Tensor(predicted_labels)), 0
                )
                # total_data = torch.cat(
                #     (total_data, data), 0
                # )
                total_outputs = torch.cat(
                    (total_outputs, y_pred.to("cpu")), 0
                )

                total_labels = torch.cat((total_labels, y_true.view(-1, 1).cpu()), 0)

        directory = f"{self.args.output_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}/epoch_{str(current_epoch)}"
        if not os.path.exists(
            f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
            )
        if not os.path.exists(
            f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
        ):
            os.makedirs(
                f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
            )
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(total_preds.cpu(), f"{directory}/preds.pt")
        torch.save(total_labels.cpu(), f"{directory}/labels.pt")
        torch.save(total_prelabels.cpu(), f"{directory}/prelabels.pt")
        torch.save(total_outputs.cpu(), f"{directory}/outputs.pt")

        total_labels = total_labels.numpy().flatten()
        total_preds = total_preds.numpy().flatten()
        total_prelabels = total_prelabels.numpy().flatten()

        precision, recall, thresholds = precision_recall_curve(
            total_labels, total_preds
        )
        test_results["auc_pr"] = metrics.auc(recall, precision)
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
                best_results_dict[key] = results_epoch[f"epoch_{str(current_epoch)}"]
                best_results_dict[key]["epochs"] = current_epoch
                # self.save_checkpoint(metrics=key)
                
                

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

        nni.report_final_result(
            self.results_test_epoch[f"epoch_{str(final_epoch)}"]["auc_pr"]
        )

    
    def save_checkpoint(self, metrics=''):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        
        
        current_epoch = self.training_state["epoch"]
        state = {
            'epoch': current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
        

        if not os.path.exists(
            f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}"
            )
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if metrics != '':
            best_path = f"{directory}/model_best_{metrics}.pth"
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        else:
            filename = f"{directory}/checkpoint-epoch{str(current_epoch)}.pth"
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))