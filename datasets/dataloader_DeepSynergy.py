from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class DeepSynergy_dataset(Dataset):
    def __init__(self, args, type):
        self.args = args
        self.type = type

        self.load_DeepDDs()

    def load_DeepDDs(self):
        if self.type == "warmup":
            self.data_dir = self.args.warmup_data_dir
        elif self.type == "noisylabel and warmup":
            self.data_dir = self.args.noisylabel_warmup_data_dir

        print(f"Run fold {str(self.args.data_partition_idx)}.")
        datafile_train = f"train_fold{str(self.args.data_partition_idx)}"
        datafile_test = f"test_fold{str(self.args.data_partition_idx)}"
        datafile_valid = f"valid_fold{str(self.args.data_partition_idx)}"

        print(f"Loading training data from {self.data_dir}")
        self.drug1_data_train = torch.load(
            os.path.join(self.data_dir, f"{datafile_train}_drug1.pt")
        )
        self.drug2_data_train = torch.load(
            os.path.join(self.data_dir, f"{datafile_train}_drug2.pt")
        )
        self.cell_data_train = torch.load(
            os.path.join(self.data_dir, f"{datafile_train}_cell.pt")
        )
        self.drug_label_train = torch.load(
            os.path.join(self.data_dir, f"{datafile_train}_label.pt")
        ).view(-1, 1)

        self.drug1_data_train[:, 0:200] = F.normalize(
            self.drug1_data_train[:, 0:200], dim=0
        )
        self.drug2_data_train[:, 0:200] = F.normalize(
            self.drug2_data_train[:, 0:200], dim=0
        )

        feats_train = torch.cat(
            [
                self.drug1_data_train,
                self.drug2_data_train,
                self.cell_data_train,
                self.drug_label_train,
            ],
            dim=1,
        )
        self.drug_data_train = feats_train

        if self.type == "warmup":
            if self.args.evaluate_valid:
                print(f"Loading valid data from {self.data_dir}")
                drug1_data_valid = torch.load(
                    os.path.join(self.data_dir, f"{datafile_valid}_drug1.pt")
                )
                drug2_data_valid = torch.load(
                    os.path.join(self.data_dir, f"{datafile_valid}_drug2.pt")
                )
                cell_data_valid = torch.load(
                    os.path.join(self.data_dir, f"{datafile_valid}_cell.pt")
                )
                drug_label_valid = torch.load(
                    os.path.join(self.data_dir, f"{datafile_valid}_label.pt")
                ).view(-1, 1)

                drug1_data_valid[:, 0:200] = F.normalize(
                    drug1_data_valid[:, 0:200], dim=0
                )
                drug2_data_valid[:, 0:200] = F.normalize(
                    drug2_data_valid[:, 0:200], dim=0
                )

                feats_valid = torch.cat(
                    [
                        drug1_data_valid,
                        drug2_data_valid,
                        cell_data_valid,
                        drug_label_valid,
                    ],
                    dim=1,
                )
                self.drug_data_valid = feats_valid

            print(f"Loading test data from {self.data_dir}")
            drug1_data_test = torch.load(
                os.path.join(self.data_dir, f"{datafile_test}_drug1.pt")
            )
            drug2_data_test = torch.load(
                os.path.join(self.data_dir, f"{datafile_test}_drug2.pt")
            )
            cell_data_test = torch.load(
                os.path.join(self.data_dir, f"{datafile_test}_cell.pt")
            )
            drug_label_test = torch.load(
                os.path.join(self.data_dir, f"{datafile_test}_label.pt")
            ).view(-1, 1)

            drug1_data_test[:, 0:200] = F.normalize(drug1_data_test[:, 0:200], dim=0)
            drug2_data_test[:, 0:200] = F.normalize(drug2_data_test[:, 0:200], dim=0)

            feats_test = torch.cat(
                [drug1_data_test, drug2_data_test, cell_data_test, drug_label_test],
                dim=1,
            )
            self.drug_data_test = feats_test

        self.get_positive_negative_number()

    def run(self, pred=[], probability=[]):
        # train，随机
        
        ori_pred = copy.deepcopy(pred)
        pred = pred[0::2]
        
        pred_clean = pred[: self.args.warmup_sample_number]
        pred_noisy = pred[self.args.warmup_sample_number :]
        
        

        pred_idx_clean_noisy = pred.nonzero()[0]
        pred_idx_clean = pred_clean.nonzero()[0]
        pred_idx_noisy = pred_noisy.nonzero()[0]
        pred_idx_noisy += self.args.warmup_sample_number

        if not self.args.noisylabel_only_noisy:
            pred_idx = pred_idx_clean_noisy
        else:
            pred_idx_clean_ones = np.arange(0, len(pred_clean), 1, dtype=int)
            pred_idx = np.concatenate((pred_idx_clean_ones, pred_idx_noisy))
            
            # ori_pred_idx = ori_pred.nonzero()[0]
            

            if self.args.co_refinement:
                tensor_data = torch.FloatTensor(probability)
                self.probalbility_dataset = TensorDataset(tensor_data)
                
                
                self.probalbility_dataset = torch.utils.data.Subset(
                    self.probalbility_dataset, pred_idx
                )

            self.drug_data_train_labeled = torch.utils.data.Subset(
                self.drug_data_train, pred_idx
            )

        self.data_state_evaluate = {
            "len_labeled_working": int(len(pred_idx)),
            "len_unlabeled_working": int(len(pred) - len(pred_idx)),
            "len_labeled": int(len(pred_idx_clean_noisy)),
            "len_unlabeled": int(len(pred) - len(pred_idx_clean_noisy)),
            "len_positive_labeled": int(
                len(np.intersect1d(pred_idx_clean_noisy, self.positive_index))
            ),
            "len_positive_unlabeled": int(
                self.positive_number
                - len(np.intersect1d(pred_idx_clean_noisy, self.positive_index))
            ),
            "len_negative_labeled": int(
                len(np.intersect1d(pred_idx_clean_noisy, self.negative_index))
            ),
            "len_negative_unlabeled": int(
                self.negative_number
                - len(np.intersect1d(pred_idx_clean_noisy, self.negative_index))
            ),
            "len_clean_labeled": int(len(pred_idx_clean)),
            "len_clean_unlabeled": int(len(pred_clean) - len(pred_idx_clean)),
            "len_clean_positive_labeled": int(
                len(np.intersect1d(pred_idx_clean, self.clean_positive_index))
            ),
            "len_clean_positive_unlabeled": int(
                self.clean_positive_number
                - len(np.intersect1d(pred_idx_clean, self.clean_positive_index))
            ),
            "len_clean_negative_labeled": int(
                len(np.intersect1d(pred_idx_clean, self.clean_negative_index))
            ),
            "len_clean_negative_unlabeled": int(
                self.clean_negative_number
                - len(np.intersect1d(pred_idx_clean, self.clean_negative_index))
            ),
            "len_noisy_labeled": int(len(pred_idx_noisy)),
            "len_noisy_unlabeled": int(len(pred_noisy) - len(pred_idx_noisy)),
            "len_noisy_positive_labeled": int(
                len(np.intersect1d(pred_idx_noisy, self.noisy_positive_index))
            ),
            "len_noisy_positive_unlabeled": int(
                self.noisy_positive_number
                - len(np.intersect1d(pred_idx_noisy, self.noisy_positive_index))
            ),
            "len_noisy_negative_labeled": int(
                len(np.intersect1d(pred_idx_noisy, self.noisy_negative_index))
            ),
            "len_noisy_negative_unlabeled": int(
                self.noisy_negative_number
                - len(np.intersect1d(pred_idx_noisy, self.noisy_negative_index))
            ),
        }

    def get_positive_negative_number(self):
        self.positive_number = 0
        self.positive_index = []
        self.negative_number = 0
        self.negative_index = []
        for i in range(len(self.drug_data_train)):
            if self.drug_data_train[i][-1] == 1:
                self.positive_number += 1
                self.positive_index.append(i)
            else:
                self.negative_number += 1
                self.negative_index.append(i)

        self.data_state = {
            "positive_number": self.positive_number,
            "positive_index": self.positive_index,
            "negative_number": self.negative_number,
            "negative_index": self.negative_index,
        }

        if self.type == "noisylabel and warmup":
            self.clean_positive_number = 0
            self.clean_positive_index = []
            self.clean_negative_number = 0
            self.clean_negative_index = []

            for i in range(0, self.args.warmup_sample_number):
                if self.drug_data_train[i][-1] == 1:
                    self.clean_positive_number += 1
                    self.clean_positive_index.append(i)
                else:
                    self.clean_negative_number += 1
                    self.clean_negative_index.append(i)

            self.noisy_positive_number = 0
            self.noisy_positive_index = []
            self.noisy_negative_number = 0
            self.noisy_negative_index = []

            for i in range(self.args.warmup_sample_number, len(self.drug_data_train)):
                if self.drug_data_train[i][-1] == 1:
                    self.noisy_positive_number += 1
                    self.noisy_positive_index.append(i)
                else:
                    self.noisy_negative_number += 1
                    self.noisy_negative_index.append(i)

            self.data_state["clean_positive_number"] = self.clean_positive_number
            self.data_state["clean_positive_index"] = self.clean_positive_index
            self.data_state["clean_negative_number"] = self.clean_negative_number
            self.data_state["clean_negative_index"] = self.clean_negative_index
            self.data_state["noisy_positive_number"] = self.noisy_positive_number
            self.data_state["noisy_positive_index"] = self.noisy_positive_index
            self.data_state["noisy_negative_number"] = self.noisy_negative_number
            self.data_state["noisy_negative_index"] = self.noisy_negative_index


class DeepSynergy_dataloader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.type = self.dataset.type

        self.drug_loader_train = DataLoader(
            self.dataset.drug_data_train,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )
        
    

        if self.type == "warmup":
            self.drug_loader_test = DataLoader(
                self.dataset.drug_data_test,
                batch_size=self.args.batch_size,
                shuffle=None,
                num_workers=4,
            )
            
            if self.args.evaluate_valid:
                self.drug_loader_valid = DataLoader(
                    self.dataset.drug_data_valid,
                    batch_size=self.args.batch_size,
                    shuffle=None,
                    num_workers=4,
                )
        elif self.type == "noisylabel and warmup":
            self.drug_loader_train_evaluate = DataLoader(
                self.dataset.drug_data_train,
                batch_size=self.args.batch_size,
                shuffle=None,
                num_workers=4,
            )

    def run(self, pred=[], probalbility=[]):
        self.dataset.run(pred=pred, probability=probalbility)

        self.drug_loader_train_labeled = DataLoader(
            self.dataset.drug_data_train_labeled,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=0
        )
        self.probability_dataloader = DataLoader(
            self.dataset.probalbility_dataset,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=0
        )
