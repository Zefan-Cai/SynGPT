
import os
import nni
import copy
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
from models.DeepSynergy import DeepSynergy
from datasets.dataloader_DeepSynergy import DeepSynergy_dataset, DeepSynergy_dataloader
from utils.utils import set_seed


class Trainer(object):
    def __init__(self, args):
        set_seed(args)
        self.args = args
        # self.args.full_experiment_name = f"{self.args.experiment_name}_fold{str(self.args.data_partition_idx)}_lr{str(self.args.learning_rate)[:8]}_epochs{str(self.args.epochs)}_warmup{str(self.args.warmup_noisylabel)}"
        self.args.full_experiment_name = f"{self.args.experiment_name}_epochs{str(self.args.epochs)}_warmup{str(self.args.warmup_noisylabel)}"
        # if self.args.noisylabel:
        #     self.args.full_experiment_name = (
        #         self.args.full_experiment_name + "_noisylabel"
        #     )
        # self.args.fold_experiment_name = (
        #     f"{self.args.experiment_name}_fold{str(self.args.data_partition_idx)}"
        # )

        # self.CE = nn.CrossEntropyLoss(reduction="none").to(self.args.device)
        # self.CEloss = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.CE = nn.functional.binary_cross_entropy_with_logits
        self.CEloss = nn.functional.binary_cross_entropy_with_logits

        self.warmup_dataset = DeepSynergy_dataset(self.args, type="warmup")
        self.warmup_data_loader = DeepSynergy_dataloader(self.args, self.warmup_dataset)

        self.warmup_drug_data_train = self.warmup_dataset.drug_data_train
        self.warmup_drug_loader_train = self.warmup_data_loader.drug_loader_train

        if self.args.evaluate_valid:
            self.warmup_drug_data_valid = self.warmup_dataset.drug_data_valid
            self.warmup_drug_loader_valid = self.warmup_data_loader.drug_loader_valid

        self.warmup_drug_data_test = self.warmup_dataset.drug_data_test
        self.warmup_drug_loader_test = self.warmup_data_loader.drug_loader_test

        self.args.warmup_sample_number = len(self.warmup_drug_data_train)

        self.noisylabel_warmup_dataset = DeepSynergy_dataset(
            self.args, type="noisylabel and warmup"
        )
        self.noisylabel_warmup_data_loader = DeepSynergy_dataloader(
            self.args, self.noisylabel_warmup_dataset
        )

        self.noisylabel_warmup_drug_data_train = (
            self.noisylabel_warmup_dataset.drug_data_train
        )
        self.noisylabel_warmup_drug_loader_train = (
            self.noisylabel_warmup_data_loader.drug_loader_train
        )

        self.noisylabel_warmup_drug_data_train_evaluate = (
            self.noisylabel_warmup_dataset.drug_data_train
        )
        self.noisylabel_warmup_drug_loader_train_evaluate = (
            self.noisylabel_warmup_data_loader.drug_loader_train_evaluate
        )

        self.model_1 = DeepSynergy().to(self.args.device)
        self.model_2 = DeepSynergy().to(self.args.device)
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

        self.warmup_best_results_test_dict = copy.deepcopy(self.best_results_test_dict)

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

        # self.args.number_train_positive = self.dataset.data_state[
        #     "number_train_positive"
        # ]
        # self.args.number_train_negative = self.dataset.data_state[
        #     "number_train_negative"
        # ]
        for key, value in vars(args).items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("\n")

    def train_epoch(self, model_index, data_loader):
        total_loss = 0
        current_epoch = self.training_state["epoch"]

        model = self.models[model_index]
        optimizer = self.optimizers[model_index]

        model.train()

        for batch_idx, data_ori in enumerate(data_loader):
            data = data_ori[:, :-1].to(self.args.device)
            drug1, drug2, cell = (
                data[:, 0 : 256 + 346 + 200],
                data[:, (256 + 346 + 200) : (256 + 346 + 200) * 2],
                data[:, (256 + 346 + 200) * 2 :],
            )
            assert drug1.size(1) == drug2.size(1)
            # pdb.set_trace()
            assert cell.size(1) == 37261
            # data_swap = torch.cat([drug2, drug1, cell], dim=1)
            # data = torch.cat([data, data_swap], dim=0)
            y = data_ori[:, -1].view(-1, 1).float().to(self.args.device)
            y = y.squeeze(1)
            # y = torch.cat([y, y])

            optimizer.zero_grad()

            output = model(data)
            
            # print(f"debug: output {output.squeeze().shape}")
            # print(f"debug: output {output.squeeze()[:10]}")
            # print(f"debug: y {y.shape}")
            # print(f"debug: y {y[:10]}")
            
            loss = self.CEloss(output.squeeze(), y.float())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

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
        

        
        iter_probalbility_loader = iter(probability_dataloader)

        model.train()

        for batch_idx, data_ori in enumerate(data_loader):
            
            # w_x = torch.FloatTensor(
            #     [
            #         probalbility_dataset[i]
            #         for i in range(batch_idx * self.args.batch_size, (batch_idx + 2) * self.args.batch_size)
            #     ],
            # ).squeeze().to(self.args.device)
            
            
            w_x = iter_probalbility_loader.next()[0]

            data = data_ori[:, :-1].to(self.args.device)
            drug1, drug2, cell = (
                data[:, 0 : 256 + 346 + 200],
                data[:, (256 + 346 + 200) : (256 + 346 + 200) * 2],
                data[:, (256 + 346 + 200) * 2 :],
            )
            
            assert drug1.size(1) == drug2.size(1)
            # pdb.set_trace()
            assert cell.size(1) == 37261
            
            # data_swap = torch.cat([drug2, drug1, cell], dim=1)
            # data = torch.cat([data, data_swap], dim=0)
            
            target = data_ori[:, -1].view(-1, 1).float().to(self.args.device)
            target = target.squeeze(1)

            # target = torch.cat([target, target])

            if self.args.co_refinement:
                
                
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)
                w_x = w_x.squeeze().to(self.args.device)

                with torch.no_grad():
                    
                    output_1 = self.model_1(data)
                    output_2 = self.model_2(data)
                    
                    px = (F.sigmoid(output_1) + F.sigmoid(output_2)) / 2
                    
                    target = w_x.squeeze() * target.squeeze() + (1 - w_x).squeeze() * px.squeeze()
                    target = target.detach()

            output = model(data)


            if self.args.penalty_loss:
                # regularization
                prior = torch.tensor(self.args.prior)
                prior = prior.cuda()
                pred_mean = F.sigmoid(output).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = self.CEloss(output.squeeze(), target.float())
            
            
            # print(f"debug: loss {loss}")
            # print(f"debug: output {output}")

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

    def eval_train(self, model_index):
        model = self.models[model_index]
        model.eval()

        losses_positive = torch.zeros(
            self.noisylabel_warmup_dataset.data_state["positive_number"]
        )
        losses_positive_index, positive_index = {}, []
        losses_negative = torch.zeros(
            self.noisylabel_warmup_dataset.data_state["negative_number"]
        )
        losses_negative_index, negative_index = {}, []

        with torch.no_grad():
            for batch_idx, data_ori in enumerate(
                self.noisylabel_warmup_drug_loader_train_evaluate
            ):
                data = data_ori[:, :-1].to(self.args.device)
                drug1, drug2, cell = (
                    data[:, 0 : 256 + 346 + 200],
                    data[:, (256 + 346 + 200) : (256 + 346 + 200) * 2],
                    data[:, (256 + 346 + 200) * 2 :],
                )
                assert drug1.size(1) == drug2.size(1)
                # pdb.set_trace()
                assert cell.size(1) == 37261
                # data_swap = torch.cat([drug2, drug1, cell], dim=1)
                # data = torch.cat([data, data_swap], dim=0)

                index = range(
                    batch_idx * self.args.batch_size,
                    (batch_idx + 2) * self.args.batch_size,
                )

                target = data_ori[:, -1].view(-1, 1).float().to(self.args.device)
                target = target.squeeze(1)
                # target = torch.cat([target, target])

                output = model(data)

                loss = self.CE(output.squeeze(), target.float(), reduction="none")

                for b in range(target.size(0)):
                    if target[b] == 1:
                        positive_index.append(1)
                        losses_positive_index[index[b]] = len(positive_index) - 1
                        losses_positive[len(positive_index) - 1] = loss[b]
                    elif target[b] == 0:
                        negative_index.append(1)
                        losses_negative_index[index[b]] = len(negative_index) - 1
                        losses_negative[len(negative_index) - 1] = loss[b]

        losses_positive = (losses_positive - losses_positive.min()) / (
            losses_positive.max() - losses_positive.min()
        )
        losses_negative = (losses_negative - losses_negative.min()) / (
            losses_negative.max() - losses_negative.min()
        )
        # self.positive_loss.append(losses_positive)
        # self.negative_loss.append(losses_negative)

        input_loss = losses_positive.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob_positive = gmm.predict_proba(input_loss)
        prob_positive = prob_positive[:, gmm.means_.argmin()]

        input_loss = losses_negative.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob_negative = gmm.predict_proba(input_loss)
        prob_negative = prob_negative[:, gmm.means_.argmin()]

        pred_positive = prob_positive > self.args.p_threshold_positive
        pred_negative = prob_negative > self.args.p_threshold_negative

        prob_all = np.zeros(len(self.noisylabel_warmup_drug_data_train_evaluate))
        pred_all = np.zeros(len(self.noisylabel_warmup_drug_data_train_evaluate))
        for key, value in losses_positive_index.items():
            prob_all[key] = prob_positive[value]
            pred_all[key] = pred_positive[value]
        for key, value in losses_negative_index.items():
            prob_all[key] = prob_negative[value]
            pred_all[key] = pred_negative[value]

        return prob_all, pred_all

    def train(self):
        self.logger.info("noisylabel")

        for epoch in range(0, self.args.epochs):
            self.training_state["epoch"] = epoch
            self.logger.info(f"epoch: {epoch}")

            if epoch < self.args.warmup_noisylabel:
                self.logger.info("warm up network 0")
                self.train_epoch(0, self.warmup_drug_loader_train)
                self.logger.info("warm up network 1")
                self.train_epoch(1, self.warmup_drug_loader_train)

                # if self.args.evaluate_valid:
                #     self.predicting("valid", self.warmup_drug_loader_valid)

                # self.predicting("test", self.warmup_drug_loader_test)
            else:
                
                if epoch == self.args.warmup_noisylabel:
                    
                    for model_index in [0, 1]:
                        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
                        best_path = f"{directory}/model_{str(model_index)}_best_{self.args.best_metrics}.pth"
                    
                        self.logger.info('Loading checkpoint: {} ...'.format(best_path))
                        checkpoint = torch.load(best_path)
                        self.models[model_index].load_state_dict(checkpoint['state_dict'])
                        self.optimizers[model_index].load_state_dict(checkpoint['optimizer'])
                
                
                prob1, pred1 = self.eval_train(0)
                prob2, pred2 = self.eval_train(1)

                self.noisylabel_warmup_data_loader.run(pred1, prob1)

                drug_data_train_labeled = self.noisylabel_warmup_dataset.drug_data_train_labeled
                drug_loader_train_labeled = self.noisylabel_warmup_data_loader.drug_loader_train_labeled
                probalbility_dataset = self.noisylabel_warmup_dataset.probalbility_dataset
                probability_dataloader = self.noisylabel_warmup_data_loader.probability_dataloader
                
                # print(f"debug: drug_data_train_labeled {len(drug_data_train_labeled)}")
                # print(f"debug: probalbility_dataset {len(probalbility_dataset)}")      
                # print(f"debug: len drug_loader_train_labeled {len(drug_loader_train_labeled)}")
                # print(f"debug: len probability_dataloader {len(probability_dataloader)}")          

                self.train_epoch_mix(1, drug_loader_train_labeled, probability_dataloader)

                self.save_GMM_state(
                    self.noisylabel_warmup_data_loader.dataset,
                    "warmup+noisylabel_mode1",
                )

                self.noisylabel_warmup_data_loader.run(pred2, prob2)

                drug_data_train_labeled = self.noisylabel_warmup_dataset.drug_data_train_labeled
                drug_loader_train_labeled = self.noisylabel_warmup_data_loader.drug_loader_train_labeled
                probalbility_dataset = self.noisylabel_warmup_dataset.probalbility_dataset
                probability_dataloader = self.noisylabel_warmup_data_loader.probability_dataloader
            
                

                self.train_epoch_mix(0, drug_loader_train_labeled, probability_dataloader)

                self.save_GMM_state(
                    self.noisylabel_warmup_data_loader.dataset,
                    "warmup+noisylabel_mode2",
                )

            if self.args.evaluate_valid:
                self.predicting("valid", self.warmup_drug_loader_valid)
            self.predicting("test", self.warmup_drug_loader_test)


            self.save_results()
            # self.save_checkpoint(model_index=0)
            # self.save_checkpoint(model_index=1)
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.console_handler)
        # self.wandb_writer.finish()

    def predicting(self, test_type, data_loader):
        self.model_1.eval()
        self.model_2.eval()

        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_prelabels = torch.Tensor()
        test_results = {}

        with torch.no_grad():
            for data_ori in data_loader:
                data = data_ori[:, :-1].to(self.args.device)
                y = data_ori[:, -1].view(-1, 1).long().to(self.args.device)

                output_1 = self.model_1(data)
                output_2 = self.model_2(data)

                outputs = output_1 + output_2
                outputs = torch.div(outputs, 2)

                # ys = F.softmax(outputs, 1).to("cpu").data.numpy()
                # predicted_labels = list(map(lambda x: np.argmax(x), ys))
                # predicted_scores = list(map(lambda x: x[1], ys))

                ys = F.sigmoid(outputs).to("cpu").data.numpy()
                predicted_labels = list(ys > 0.5)
                predicted_scores = list(ys)

                total_preds = torch.cat(
                    (total_preds, torch.Tensor(predicted_scores)), 0
                )
                total_prelabels = torch.cat(
                    (total_prelabels, torch.Tensor(predicted_labels)), 0
                )
                total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)

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

        if self.training_state["epoch"] < self.args.warmup_noisylabel:
            best_results_dict = self.warmup_best_results_test_dict
            for key in ["auc_pr", "max_F1", "bal_acc", "kappa_score"]:
                if (
                    results_epoch[f"epoch_{str(current_epoch)}"][key]
                    > best_results_dict[key][key]
                ):
                    best_results_dict[key] = results_epoch[
                        f"epoch_{str(current_epoch)}"
                    ]
                    best_results_dict[key]["epochs"] = current_epoch

                    # self.save_checkpoint(metrics=key, model_index=0, type="warmup")
                    # self.save_checkpoint(metrics=key, model_index=1, type="warmup")

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

                    self.save_checkpoint(metrics=key, model_index=0, type="noisylabel")
                    self.save_checkpoint(metrics=key, model_index=1, type="noisylabel")

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

        with open(directory + "/warmup_best_results_test_dict.json", "w") as fp:
            json.dump(self.warmup_best_results_test_dict, fp)

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

    def save_GMM_state(self, dataset, info):
        current_epoch = self.training_state["epoch"]
        self.GMM_state_epoch[
            f"epoch_{str(current_epoch)}_{info}"
        ] = dataset.data_state_evaluate
        for key, value in self.GMM_state_epoch[
            f"epoch_{str(current_epoch)}_{info}"
        ].items():
            self.logger.info(f"{key}: {value}")

    def save_checkpoint(self, metrics="", model_index=0, type=""):
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
            "optimizer": optimizer.state_dict(),
        }
        # if save_best:

        directory = f"{self.args.checkpoint_dir}/fold{str(self.args.data_partition_idx)}/{self.args.full_experiment_name}"
        if type != "":
            directory += f"_{type}"


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
            self.logger.info(
                f"Saving epoch {str(current_epoch)} checkpoint: {filename}"
            )
