
import os
import nni
import json
import numpy as np

from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    precision_recall_curve,
    cohen_kappa_score,
    balanced_accuracy_score,
    classification_report,
    auc,
)

from torch.utils.tensorboard import SummaryWriter

from utils.log import my_logger, get_logger
from datasets.dataloader_AuDNNsynergy import (
    AuDNNsynergy_dataset,
    AuDNNsynergy_dataloader
)
from models.AuDNNsynergy import DNN
from utils.utils import set_seed


class Trainer(object):
    def __init__(self, args):
        set_seed(args)
        self.args = args
        self.args.full_experiment_name = f"{self.args.experiment_name}_epochs{str(self.args.epochs)}_warmup{str(self.args.warmup_noisylabel)}"
        # if self.args.noisylabel:
        #     self.args.full_experiment_name = (
        #         self.args.full_experiment_name + "_noisylabel"
        #     )
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
        
        self.args.warmup_sample_number = len(self.warmup_dataset_train)
        
        # warmup
        self.noisylabel_warmup_dataset = AuDNNsynergy_dataset(self.args, type="noisylabel and warmup")
        self.noisylabel_warmup_dataloader = AuDNNsynergy_dataloader(self.args, self.noisylabel_warmup_dataset)
        
        self.noisylabel_warmup_dataset_train = self.noisylabel_warmup_dataset.data_train
        self.noisylabel_warmup_dataloader_train = self.noisylabel_warmup_dataloader.loader_train
        self.noisylabel_warmup_dataloader_train_evaluate = self.noisylabel_warmup_dataloader.loader_train_evaluate
        
        
        
        self.args.data_state = {}
        for key, item in self.noisylabel_warmup_dataset.data_state.items():
            if "number" in key:
                self.args.data_state[key] = item


        self.model_1 = DNN(self.warmup_dataset_train.cell_feat_len() + 2 * self.warmup_dataset_train.drug_feat_len(), self.args.hidden_size).to(self.args.device)
        self.model_2 = DNN(self.warmup_dataset_train.cell_feat_len() + 2 * self.warmup_dataset_train.drug_feat_len(), self.args.hidden_size).to(self.args.device)
        self.models = [self.model_1, self.model_2]

        self.params_1 = [{"params": self.model_1.parameters()}]
        self.params_2 = [{"params": self.model_2.parameters()}]

        self.optimizer_1 = torch.optim.Adam(self.params_1, lr=args.learning_rate)
        self.optimizer_2 = torch.optim.Adam(self.params_2, lr=args.learning_rate)
        self.optimizers = [self.optimizer_1, self.optimizer_2]


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

    def train_epoch(self, model_index, data_loader):
        total_loss = 0
        current_epoch = self.training_state["epoch"]

        model = self.models[model_index]
        optimizer = self.optimizers[model_index]

        model.train()

        for batch_idx, batch in enumerate(data_loader):
            
            drug1_feats, drug2_feats, cell_feats, y_true = batch
            drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.to(self.args.device), drug2_feats.to(self.args.device), cell_feats.to(self.args.device), y_true.to(self.args.device)
            y_pred = model(drug1_feats, drug2_feats, cell_feats)
            loss = self.CEloss(y_pred, y_true)

            self.training_state["batch_idx"] = batch_idx

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    def train_epoch_mix(self, model_index, data_loader, probability_dataloader):
        total_loss = 0
        current_epoch = self.training_state["epoch"]

        model = self.models[model_index]
        optimizer = self.optimizers[model_index]

        model.train()

        for batch_idx, (batch, w_x) in enumerate(zip(data_loader, probability_dataloader)):

            drug1_feats, drug2_feats, cell_feats, y_true = batch
            drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.to(self.args.device), drug2_feats.to(self.args.device), cell_feats.to(self.args.device), y_true.to(self.args.device)
            target = y_true
            w_x = w_x[0]

            if self.args.co_refinement:
                w_x = w_x.view(-1, 1).type(torch.FloatTensor).squeeze()
                w_x = w_x.to(self.args.device)

                with torch.no_grad():
                    y_pred_1 = self.model_1(drug1_feats, drug2_feats, cell_feats)
                    y_pred_2 = self.model_2(drug1_feats, drug2_feats, cell_feats)
                    px = (F.sigmoid(y_pred_1) + F.sigmoid(y_pred_2)) / 2

                    
                    target = w_x.squeeze() * target.squeeze() + (1 - w_x).squeeze() * px.squeeze()
                    target = target.detach()

            y_pred = model(drug1_feats, drug2_feats, cell_feats)

            if self.args.penalty_loss:
                # regularization
                prior = torch.tensor(self.args.prior)
                prior = prior.cuda()
                pred_mean = F.sigmoid(y_pred).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

        
            loss = self.CEloss(y_pred.squeeze(), target.squeeze())
            if self.args.penalty_loss:
                loss = loss + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            self.training_state["step"] += 1
            self.training_state["batch_idx"] = batch_idx

        total_loss /= len(data_loader)

        self.logger.info(f"loss: {total_loss}")
        self.writer.add_scalars(
            f"loss/loss",
            {f"loss": total_loss},
            self.training_state["epoch"],
        )
        # self.wandb_writer.log({f"loss": total_loss}, step=self.training_state["epoch"])
        self.loss_epoch[self.training_state["epoch"]] = total_loss

    def eval_train(self, model_index, evaluate_data_loader):
        model = self.models[model_index]
        model.eval()

        losses_positive = torch.zeros(
            evaluate_data_loader.dataset.data_state["positive_number"]
        )
        losses_positive_index, positive_index = {}, []
        losses_negative = torch.zeros(
            evaluate_data_loader.dataset.data_state["negative_number"]
        )
        losses_negative_index, negative_index = {}, []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                evaluate_data_loader.loader_train_evaluate
            ):
                drug1_feats, drug2_feats, cell_feats, y_true = batch
                drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.to(self.args.device), drug2_feats.to(self.args.device), cell_feats.to(self.args.device), y_true.to(self.args.device)
                target = y_true
                
                index = range(
                    batch_idx * self.args.batch_size,
                    (batch_idx + 1) * self.args.batch_size,
                )

                y_pred = model(drug1_feats, drug2_feats, cell_feats)
                loss = self.CEloss(y_pred, y_true, reduction="none")

                for b in range(target.size(0)):
                    if target[b] == 1:
                        positive_index.append(1)
                        losses_positive_index[index[b]] = len(positive_index) - 1
                        losses_positive[len(positive_index) - 1] = loss[b]
                    elif target[b] == 0:
                        negative_index.append(1)
                        losses_negative_index[index[b]] = len(negative_index) - 1
                        losses_negative[len(negative_index) - 1] = loss[b]


        prob_all = np.zeros(len(evaluate_data_loader.data_train))
        pred_all = np.zeros(len(evaluate_data_loader.data_train))

        losses_positive = (losses_positive - losses_positive.min()) / (
            losses_positive.max() - losses_positive.min()
        )
        input_loss = losses_positive.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob_positive = gmm.predict_proba(input_loss)
        prob_positive = prob_positive[:, gmm.means_.argmin()]

        pred_positive = prob_positive > self.args.p_threshold_positive

        for key, value in losses_positive_index.items():
            prob_all[key] = prob_positive[value]
            pred_all[key] = pred_positive[value]

        if evaluate_data_loader.dataset.data_state["negative_number"] != 0:
            losses_negative = (losses_negative - losses_negative.min()) / (
                losses_negative.max() - losses_negative.min()
            )

            input_loss = losses_negative.reshape(-1, 1)

            # fit a two-component GMM to the loss
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_loss)
            prob_negative = gmm.predict_proba(input_loss)
            prob_negative = prob_negative[:, gmm.means_.argmin()]

            if not self.args.negative_GMM:
                prob_negative = [1 for i in range(len(prob_negative))]

            pred_negative = prob_negative > self.args.p_threshold_negative

            for key, value in losses_negative_index.items():
                prob_all[key] = prob_negative[value]
                pred_all[key] = pred_negative[value]

        return prob_all, pred_all

    def train(self):
        self.logger.info("noisylabel")
        # self.wandb_writer = wandb.init(project=self.args.fold_experiment_name+"_noisylabel", name=self.args.full_experiment_name+"_noisylabel", config=self.args)
        # self.wandb_writer.watch(self.model_1)
        # self.wandb_writer.watch(self.model_2)

        for epoch in range(0, self.args.epochs):
            self.training_state["epoch"] = epoch
            self.logger.info(f"epoch: {epoch}")

            if epoch < self.args.warmup_noisylabel:
                self.logger.info("warm up network 0 with warmup dataset")
                self.train_epoch(0, self.warmup_dataloader_train)
                self.logger.info("warm up network 1 with warmup dataset")
                self.train_epoch(1, self.warmup_dataloader_train)

            else:
                
                if epoch == self.args.warmup_noisylabel:
                    
                    for model_index in [0, 1]:
                        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
                        best_path = f"{directory}/model_{str(model_index)}_best_{self.args.best_metrics}.pth"
                    
                        self.logger.info('Loading checkpoint: {} ...'.format(best_path))
                        checkpoint = torch.load(best_path)
                        self.models[model_index].load_state_dict(checkpoint['state_dict'])
                        self.optimizers[model_index].load_state_dict(checkpoint['optimizer'])
                
                prob1, pred1 = self.eval_train(
                    0, self.noisylabel_warmup_dataloader
                )
                prob2, pred2 = self.eval_train(
                    1, self.noisylabel_warmup_dataloader
                )
                    
                self.noisylabel_warmup_dataloader.run(pred1, prob1)
                dataloader_train_labeled = self.noisylabel_warmup_dataloader.dataloader_train_labeled
                probability_dataloader = self.noisylabel_warmup_dataloader.probability_dataloader
                
                self.save_GMM_state(
                    self.noisylabel_warmup_dataset, "warmup+noisylabel_model1"
                )

                
                self.train_epoch_mix(1, dataloader_train_labeled, probability_dataloader)

                self.noisylabel_warmup_dataloader.run(pred2, prob2)
                probability_dataloader = self.noisylabel_warmup_dataloader.probability_dataloader
                dataloader_train_labeled = self.noisylabel_warmup_dataloader.dataloader_train_labeled
                self.save_GMM_state(
                    self.noisylabel_warmup_dataset, "warmup+noisylabel_mode2"
                )

                self.train_epoch_mix(0, dataloader_train_labeled, probability_dataloader)

            if self.args.evaluate_valid:
                self.predicting("valid", self.dataloader_valid)
            self.predicting("test", self.dataloader_test)
            self.save_results()
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.console_handler)

    def predicting(self, test_type, data_loader):
        self.model_1.eval()
        self.model_2.eval()

        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_prelabels = torch.Tensor()
        test_results = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                
                drug1_feats, drug2_feats, cell_feats, y_true = batch
                drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.to(self.args.device), drug2_feats.to(self.args.device), cell_feats.to(self.args.device), y_true.to(self.args.device)
                

                y_pred_1 = self.model_1(drug1_feats, drug2_feats, cell_feats)
                y_pred_2 = self.model_2(drug1_feats, drug2_feats, cell_feats)
                y_pred = (y_pred_1 + y_pred_2) / 2
                ys = F.sigmoid(y_pred).to("cpu").data.numpy()

                predicted_labels = list(ys > 0.5)
                predicted_scores = list(ys)
                total_preds = torch.cat(
                    (total_preds, torch.Tensor(predicted_scores)), 0
                )
                total_prelabels = torch.cat(
                    (total_prelabels, torch.Tensor(predicted_labels)), 0
                )
                total_labels = torch.cat((total_labels, y_true.view(-1, 1).cpu()), 0)

        total_labels = total_labels.numpy().flatten()
        total_preds = total_preds.numpy().flatten()
        total_prelabels = total_prelabels.numpy().flatten()

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

        if self.training_state["epoch"] >= self.args.warmup_noisylabel:
            for key in ["auc_pr", "max_F1", "bal_acc", "kappa_score"]:
                
                if self.training_state["epoch"] == self.args.warmup_noisylabel:
                    best_results_dict[key][key] = 0
                
                if (
                    results_epoch[f"epoch_{str(current_epoch)}"][key]
                    > best_results_dict[key][key]
                ):
                    best_results_dict[key] = results_epoch[
                        f"epoch_{str(current_epoch)}"
                    ]
                    best_results_dict[key]["epochs"] = current_epoch
                    self.save_checkpoint(metrics=key, model_index=0)
                    self.save_checkpoint(metrics=key, model_index=1)
        else:
            for key in ["auc_pr", "max_F1", "bal_acc", "kappa_score"]:
                if (
                    results_epoch[f"epoch_{str(current_epoch)}"][key]
                    > best_results_dict[key][key]
                ):
                    best_results_dict[key] = results_epoch[
                        f"epoch_{str(current_epoch)}"
                    ]
                    best_results_dict[key]["epochs"] = current_epoch
                    self.save_checkpoint(metrics=key, model_index=0)
                    self.save_checkpoint(metrics=key, model_index=1)

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
        
        
    def save_GMM_state(self, data_loader, info):
        current_epoch = self.training_state["epoch"]
        self.GMM_state_epoch[
            f"epoch_{str(current_epoch)}_{info}"
        ] = data_loader.data_state_evaluate
        print(data_loader.data_state_evaluate)
        for key, value in self.GMM_state_epoch[
            f"epoch_{str(current_epoch)}_{info}"
        ].items():
            self.logger.info(f"{key}: {value}")

        # for key in data_loader.data_state_evaluate.keys():
        #     self.writer.add_scalars(
        #         f"GMM_state/{key}",
        #         {"noisylabel": data_loader.data_state_evaluate[key]},
        #         self.training_state["epoch"],
        #     )
        # self.wandb_writer.log(self.dataset.data_state_evaluate, step=self.training_state["epoch"])

    def save_checkpoint(self, metrics="", model_index=0):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        model = self.models[model_index]
        optimizer = self.optimizers[model_index]
        current_epoch = self.training_state["epoch"]
        arch = type(model).__name__
        state = {
            "arch": arch,
            "epoch": current_epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # 'monitor_best': self.mnt_best,
            # 'config': self.config
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
            best_path = f"{directory}/model_{str(model_index)}_best_{metrics}.pth"
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        else:
            filename = f"{directory}/checkpoint_{str(model_index)}-epoch{str(current_epoch)}.pth"
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
