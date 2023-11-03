
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

# 我的package
from utils.log import my_logger, get_logger
from datasets.dataloader_GraphSynergy import (
    GraphSynergy_dataloader,
    GraphSynergy_dataset,
    GraphSynergy_dataset_info,
)
from models.GraphSynergy import GraphSynergy as module_arch
from utils.utils import set_seed


class Trainer(object):
    def __init__(self, args):
        set_seed(args)
        self.args = args
        self.args.full_experiment_name = f"{self.args.experiment_name}_epochs{str(self.args.epochs)}_warmup{str(self.args.warmup_noisylabel)}"
        if self.args.noisylabel:
            self.args.full_experiment_name = (
                self.args.full_experiment_name + "_noisylabel"
            )
        self.CE = nn.functional.binary_cross_entropy_with_logits
        self.CEloss = nn.functional.binary_cross_entropy_with_logits

        self.dataset = GraphSynergy_dataset_info(self.args)

        # warmup
        self.warmup_data_loader = GraphSynergy_dataloader(
            self.args, self.dataset, type="warmup", valid=True, test=True
        )

        self.warmup_train_dataset = self.warmup_data_loader.train_dataset
        self.valid_dataset = self.warmup_data_loader.valid_dataset
        self.test_dataset = self.warmup_data_loader.test_dataset

        self.args.warmup_sample_number = len(self.warmup_train_dataset)

        self.warmup_train_data_loader = self.warmup_data_loader.train_data_loader
        self.warmup_evaluate_data_loader = self.warmup_data_loader.evaluate_data_loader
        self.valid_data_loader = self.warmup_data_loader.valid_data_loader
        self.test_data_loader = self.warmup_data_loader.test_data_loader

        self.noisylabel_warmup_data_loader = GraphSynergy_dataloader(
            self.args, self.dataset, type="noisylabel and warmup"
        )
        self.noisylabel_warmup_train_dataset = (
            self.noisylabel_warmup_data_loader.train_dataset
        )
        self.noisylabel_warmup_train_data_loader = (
            self.noisylabel_warmup_data_loader.train_data_loader
        )
        self.noisylabel_warmup_evaluate_data_loader = (
            self.noisylabel_warmup_data_loader.evaluate_data_loader
        )

        self.noisylabel_warmup_softlabel_data_loader_0 = GraphSynergy_dataloader(
            self.args, self.dataset, type="noisylabel and warmup"
        )
        self.noisylabel_warmup_softlabel_train_dataset = (
            self.noisylabel_warmup_softlabel_data_loader_0.train_dataset
        )
        self.noisylabel_warmup__softlabeltrain_data_loader = (
            self.noisylabel_warmup_softlabel_data_loader_0.train_data_loader
        )

        self.noisylabel_warmup_softlabel_data_loader_1 = GraphSynergy_dataloader(
            self.args, self.dataset, type="noisylabel and warmup"
        )
        self.noisylabel_warmup_softlabel_train_dataset = (
            self.noisylabel_warmup_softlabel_data_loader_1.train_dataset
        )
        self.noisylabel_warmup__softlabeltrain_data_loader = (
            self.noisylabel_warmup_softlabel_data_loader_1.train_data_loader
        )

        self.args.data_state = {}
        for key, item in self.noisylabel_warmup_data_loader.data_state.items():
            if "number" in key:
                self.args.data_state[key] = item

        self.feature_index = self.dataset.get_feature_index()
        self.cell_neighbor_set = self.dataset.get_cell_neighbor_set()
        self.drug_neighbor_set = self.dataset.get_drug_neighbor_set()
        self.node_num_dict = self.dataset.get_node_num_dict()

        self.model_1 = module_arch(self.args, self.node_num_dict)
        self.model_2 = module_arch(self.args, self.node_num_dict)
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

        for batch_idx, (
            data,
            target,
        ) in enumerate(data_loader):
            data = data.to(self.args.device)
            target = target.to(self.args.device)

            output, emb_loss = model(*self._get_feed_dict(data))

            try:
                loss = self.CEloss(output, target.squeeze()) + emb_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except:
                self.logger.warning(f"Error: {output.shape} {target.shape}")

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











    

























    def get_SoftLabel(self, data_loader):
        self.targets = []

        for batch_idx, (data, target, w_x) in enumerate(data_loader):
            data = data.to(self.args.device)
            target = target.to(self.args.device)

            if self.args.co_refinement:
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)
                w_x = w_x.to(self.args.device)

                with torch.no_grad():
                    output_1, emb_loss = self.model_1(*self._get_feed_dict(data))
                    output_2, emb_loss = self.model_2(*self._get_feed_dict(data))
                    px = (F.sigmoid(output_1) + F.sigmoid(output_2)) / 2
                    px = w_x.squeeze() * target.squeeze() + (1 - w_x).squeeze() * px

                    if self.args.SoftLabel == "CoRefine":
                        self.targets.append(px.detach())

                    px1 = 1 - px
                    alpha = 1.1  # >1
                    px_alpha = px**self.args.Sharpen_Alpha
                    px1_alpha = px1**self.args.Sharpen_Alpha
                    px_alpha_sum = px_alpha + px1_alpha
                    targets_x = px / px_alpha_sum
                    targets_x = targets_x.detach()

                    if self.args.SoftLabel == "Sharpen":
                        self.targets.append(targets_x.detach())

        self.targets = torch.cat(self.targets, dim=0)

    def train_epoch_mix(self, model_index, data_loader):
        total_loss = 0
        current_epoch = self.training_state["epoch"]

        model = self.models[model_index]
        optimizer = self.optimizers[model_index]

        self.targets = []

        model.train()

        for batch_idx, (data, target, w_x) in enumerate(data_loader):
            data = data.to(self.args.device)
            target = target.to(self.args.device)

            if self.args.co_refinement:
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)
                w_x = w_x.to(self.args.device)

                with torch.no_grad():
                    output_1, emb_loss = self.model_1(*self._get_feed_dict(data))
                    output_2, emb_loss = self.model_2(*self._get_feed_dict(data))
                    px = (F.sigmoid(output_1) + F.sigmoid(output_2)) / 2
                    px = w_x.squeeze() * target.squeeze() + (1 - w_x).squeeze() * px

                    if self.args.SoftLabel == "CoRefine":
                        self.targets.append(px.detach())

                    if self.args.Sharpen:
                        px1 = 1 - px
                        alpha = 1.1  # >1
                        px_alpha = px**self.args.Sharpen_Alpha
                        px1_alpha = px1**self.args.Sharpen_Alpha
                        px_alpha_sum = px_alpha + px1_alpha
                        targets_x = px / px_alpha_sum
                        targets_x = targets_x.detach()
                    else:
                        targets_x = px.detach()

                    if self.args.SoftLabel == "Sharpen":
                        self.targets.append(targets_x.detach())

            output, emb_loss = model(*self._get_feed_dict(data))

            if self.args.penalty_loss:
                # regularization
                prior = torch.tensor(self.args.prior)
                prior = prior.cuda()
                pred_mean = F.sigmoid(output).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

            try:
                loss = self.CEloss(output, targets_x.squeeze()) + emb_loss
                if self.args.penalty_loss:
                    loss = loss + penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except:
                self.logger.warning(f"Error: {output.shape} {target.shape}")

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
            evaluate_data_loader.data_state["positive_number"]
        )
        losses_positive_index, positive_index = {}, []
        losses_negative = torch.zeros(
            evaluate_data_loader.data_state["negative_number"]
        )
        losses_negative_index, negative_index = {}, []

        targets = []
        losses = []
        outputs = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(
                evaluate_data_loader.evaluate_data_loader
            ):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                index = range(
                    batch_idx * self.args.batch_size,
                    (batch_idx + 1) * self.args.batch_size,
                )

                output, emb_loss = model(*self._get_feed_dict(data))
                loss = self.CE(output, target.squeeze(), reduction="none") + emb_loss

                for b in range(target.size(0)):
                    if target[b] == 1:
                        positive_index.append(1)
                        losses_positive_index[index[b]] = len(positive_index) - 1
                        losses_positive[len(positive_index) - 1] = loss[b]
                    elif target[b] == 0:
                        negative_index.append(1)
                        losses_negative_index[index[b]] = len(negative_index) - 1
                        losses_negative[len(negative_index) - 1] = loss[b]

                targets.append(target)
                losses.append(loss)
                outputs.append(output)

        # self.positive_loss.append(losses_positive)
        # self.negative_loss.append(losses_negative)

        prob_all = np.zeros(len(evaluate_data_loader.train_dataset))
        pred_all = np.zeros(len(evaluate_data_loader.train_dataset))

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

        if evaluate_data_loader.data_state["negative_number"] != 0:
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

        self.prob_all = prob_all
        self.targets_share = torch.cat(targets, dim=0).cpu().numpy()
        self.losses = torch.cat(losses, dim=0).cpu().numpy()
        self.outputs = torch.cat(outputs, dim=0).cpu().numpy()


        return prob_all, pred_all, self.outputs











































    def train(self):
        self.logger.info("noisylabel")
        # self.wandb_writer = wandb.init(project=self.args.fold_experiment_name+"_noisylabel", name=self.args.full_experiment_name+"_noisylabel", config=self.args)
        # self.wandb_writer.watch(self.model_1)
        # self.wandb_writer.watch(self.model_2)

        for epoch in range(0, self.args.epochs):
            self.training_state["epoch"] = epoch
            self.logger.info(f"epoch: {epoch}")

            if epoch < self.args.warmup_noisylabel:
                # if self.args.warmup_dataset == "warmup":
                self.logger.info("warm up network 0 with warmup dataset")
                self.train_epoch(0, self.warmup_train_data_loader)
                self.logger.info("warm up network 1 with warmup dataset")
                self.train_epoch(1, self.warmup_train_data_loader)
                # elif self.args.warmup_dataset == "warmup+noisylabel":
                #     self.logger.info(
                #         "warm up network 0 with warmup + noisylabel dataset"
                #     )
                #     self.train_epoch(0, self.noisylabel_warmup_train_data_loader)
                #     self.logger.info(
                #         "warm up network 1 with warmup + noisylabel dataset"
                #     )
                #     self.train_epoch(1, self.noisylabel_warmup_train_data_loader)
                if self.args.evaluate_valid:
                    self.predicting("valid", self.valid_data_loader)
                self.predicting("test", self.test_data_loader)
            else:
                if (
                    self.args.ReloadCkpt == "first"
                    and epoch == self.args.warmup_noisylabel
                ):
                    for model_index in [0, 1]:
                        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
                        best_path = f"{directory}/model_{str(model_index)}_best_{self.args.best_metrics}.pth"
                        self.logger.info("Loading checkpoint: {} ...".format(best_path))
                        checkpoint = torch.load(best_path)
                        self.models[model_index].load_state_dict(
                            checkpoint["state_dict"]
                        )
                        self.optimizers[model_index].load_state_dict(
                            checkpoint["optimizer"]
                        )
                elif self.args.ReloadCkpt == "every":
                    for model_index in [0, 1]:
                        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
                        best_path = f"{directory}/model_{str(model_index)}_best_{self.args.best_metrics}.pth"
                        self.logger.info("Loading checkpoint: {} ...".format(best_path))
                        checkpoint = torch.load(best_path)
                        self.models[model_index].load_state_dict(
                            checkpoint["state_dict"]
                        )
                        self.optimizers[model_index].load_state_dict(
                            checkpoint["optimizer"]
                        )
                elif self.args.ReloadCkpt == "none":
                    pass
                else:
                    raise ValueError("ReloadBestPerEoch must be first, every or none")

                prob0, pred0, output0 = self.eval_train(0, self.noisylabel_warmup_data_loader)
                prob1, pred1, output1 = self.eval_train(1, self.noisylabel_warmup_data_loader)

                self.noisylabel_warmup_softlabel_data_loader_0.run(
                    pred0, prob0, drop=False
                )
                train_softlabel_dataset_labeled = (
                    self.noisylabel_warmup_softlabel_data_loader_0.train_dataset_labeled
                )
                train_softlabel_data_loader_labeled = (
                    self.noisylabel_warmup_softlabel_data_loader_0.data_loader_labeled
                )

                if self.args.SoftLabel != "None":
                    self.get_SoftLabel(train_softlabel_data_loader_labeled)
                else:
                    self.targets = None

                self.noisylabel_warmup_data_loader.run(
                    pred0, prob0, self.targets, drop=True
                )
                train_dataset_labeled = (
                    self.noisylabel_warmup_data_loader.train_dataset_labeled
                )
                train_data_loader_labeled = (
                    self.noisylabel_warmup_data_loader.data_loader_labeled
                )

                self.save_GMM_state(
                    self.noisylabel_warmup_data_loader, "warmup+noisylabel_model1"
                )
                self.train_epoch_mix(1, train_data_loader_labeled)

                if self.args.save_data == True:
                    self.save_data(0)

                self.noisylabel_warmup_softlabel_data_loader_1.run(
                    pred1, prob1, drop=False
                )
                train_softlabel_dataset_labeled = (
                    self.noisylabel_warmup_softlabel_data_loader_1.train_dataset_labeled
                )
                train_softlabel_data_loader_labeled = (
                    self.noisylabel_warmup_softlabel_data_loader_1.data_loader_labeled
                )

                if self.args.SoftLabel != "None":
                    self.get_SoftLabel(train_softlabel_data_loader_labeled)
                else:
                    self.targets = None

                self.noisylabel_warmup_data_loader.run(
                    pred1, prob1, self.targets, drop=True
                )
                train_dataset_labeled = (
                    self.noisylabel_warmup_data_loader.train_dataset_labeled
                )
                train_data_loader_labeled = (
                    self.noisylabel_warmup_data_loader.data_loader_labeled
                )

                self.save_GMM_state(
                    self.noisylabel_warmup_data_loader, "warmup+noisylabel_mode2"
                )
                self.train_epoch_mix(0, train_data_loader_labeled)

                if self.args.save_data == True:
                    self.save_data(1)

                if self.args.evaluate_valid:
                    self.predicting("valid", self.valid_data_loader)
                self.predicting("test", self.test_data_loader)
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
            for batch_idx, (data, target) in enumerate(data_loader):
                target = target.to(self.args.device)
                output_1, emb_loss = self.model_1(*self._get_feed_dict(data))
                output_2, emb_loss = self.model_2(*self._get_feed_dict(data))

                outputs = output_1 + output_2
                outputs = torch.div(outputs, 2)
                # outputs = (output_1 + output_2) / 2

                ys = F.sigmoid(outputs).to("cpu").data.numpy()
                predicted_labels = list(ys > 0.5)
                predicted_scores = list(ys)
                total_preds = torch.cat(
                    (total_preds, torch.Tensor(predicted_scores)), 0
                )
                total_prelabels = torch.cat(
                    (total_prelabels, torch.Tensor(predicted_labels)), 0
                )
                total_labels = torch.cat((total_labels, target.view(-1, 1).cpu()), 0)

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
                    # self.save_checkpoint(metrics=key, model_index=0)
                    # self.save_checkpoint(metrics=key, model_index=1)
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

        # if test_type == "test" and self.args.evaluate_valid:
        #     for key in ["auc_pr", "max_F1", "bal_acc", "kappa_score"]:
        #         if (
        #             self.results_valid_epoch[f"epoch_{str(current_epoch)}"][key]
        #             > self.best_results_valid_dict[key][key]
        #         ):
        # self.best_results_test_valid_dict[key] = results_epoch[f"epoch_{str(current_epoch)}"]
        # self.best_results_test_valid_dict[key]["epochs"] = current_epoch

    def save_data(self, model_index):
        directory = f"{self.args.output_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        output_data = self.noisylabel_warmup_data_loader.output_data
        epoch = self.training_state["epoch"]

        np.savez(
            directory + f"/epoch_{str(epoch)}_model{str(model_index)}",
            probability=output_data[0],
            prob_all=self.prob_all,
            label=output_data[1],
            targets=self.targets_share,
            losses=self.losses,
            outputs=self.outputs
        )







    def save_results(self):
        # 设定地址
        directory = f"{self.args.result_path}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
        # 如果文件夹不存在
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 如果上级文件夹不存在
        if not os.path.exists(
            f"{self.args.result_path}/fold{str(self.args.data_partition_idx)}"
        ):
            os.makedirs(
                f"{self.args.result_path}/fold{str(self.args.data_partition_idx)}"
            )
        # 如果已经存在日志
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

    def _get_feed_dict(self, data):
        # [batch_size]
        cells = data[:, self.feature_index["cell"]]
        drugs1 = data[:, self.feature_index["drug1"]]
        drugs2 = data[:, self.feature_index["drug2"]]
        cells_neighbors, drugs1_neighbors, drugs2_neighbors = [], [], []
        for hop in range(self.model_1.n_hop):
            # try:
            cells_neighbors.append(
                torch.LongTensor(
                    [self.cell_neighbor_set[c][hop] for c in cells.cpu().numpy()]
                ).to(self.args.device)
            )
            drugs1_neighbors.append(
                torch.LongTensor(
                    [self.drug_neighbor_set[d][hop] for d in drugs1.cpu().numpy()]
                ).to(self.args.device)
            )
            drugs2_neighbors.append(
                torch.LongTensor(
                    [self.drug_neighbor_set[d][hop] for d in drugs2.cpu().numpy()]
                ).to(self.args.device)
            )
            # except:
            #     for d in drugs2.cpu().numpy():
            #         print(d)
            #         print(hop)
            #         print(self.drug_neighbor_set)

        return (
            cells.to(self.args.device),
            drugs1.to(self.args.device),
            drugs2.to(self.args.device),
            cells_neighbors,
            drugs1_neighbors,
            drugs2_neighbors,
        )

    def save_GMM_state(self, data_loader, info):
        current_epoch = self.training_state["epoch"]
        self.GMM_state_epoch[
            f"epoch_{str(current_epoch)}_{info}"
        ] = data_loader.data_state_evaluate
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

        # 如果上级文件夹不存在
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
            self.logger.info(f"Saving current best {metrics}: model_best.pth ...")
        else:
            filename = f"{directory}/checkpoint_{str(model_index)}-epoch{str(current_epoch)}.pth"
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
