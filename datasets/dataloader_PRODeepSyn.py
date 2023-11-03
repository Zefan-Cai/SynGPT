import numpy as np
import torch

import random
import os

from torch.utils.data import Dataset, TensorDataset, DataLoader

class PRODeepSyn_dataset(Dataset):
    def __init__(self, args, type=""):
        self.args = args
        self.type = type

        self.load_DeepDDs()

    def load_DeepDDs(self):


        self.data_train = FastSynergyDataset(self.args, self.type, use_folds=[0])
        self.data_test = FastSynergyDataset(self.args, self.type, use_folds=[1])    

        if self.type == "warmup":
            if self.args.evaluate_valid:
                self.data_valid = FastSynergyDataset(self.args, self.data_dir, use_folds=[1])            

        self.get_positive_negative_number()

    def run(self, pred=[], probability=[]):
        pred_clean = pred[:self.args.warmup_sample_number]
        pred_noisy = pred[self.args.warmup_sample_number:]

        pred_idx_clean_noisy = pred.nonzero()[0]
        pred_idx_clean = pred_clean.nonzero()[0]
        pred_idx_noisy = pred_noisy.nonzero()[0]
        pred_idx_noisy += self.args.warmup_sample_number

        if not self.args.noisylabel_only_noisy:
            pred_idx = pred_idx_clean_noisy
        else:
            pred_idx_clean_ones = np.arange(0, len(pred_clean), 1, dtype=int)
            pred_idx = np.concatenate((pred_idx_clean_ones, pred_idx_noisy))
            
            self.pred_idx = pred_idx

            if self.args.co_refinement:

                tensor_data = torch.tensor(probability)
                self.probalbility_dataset = TensorDataset(tensor_data)
                self.probalbility_dataset = torch.utils.data.Subset(
                    self.probalbility_dataset, pred_idx
                )

            self.data_train_labeled = torch.utils.data.Subset(
                self.data_train, pred_idx
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
        for i in range(len(self.data_train)):
            if self.data_train.samples[i][3] == 1:
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
                if self.data_train.samples[i][3] == 1:
                    self.clean_positive_number += 1
                    self.clean_positive_index.append(i)
                else:
                    self.clean_negative_number += 1
                    self.clean_negative_index.append(i)

            self.noisy_positive_number = 0
            self.noisy_positive_index = []
            self.noisy_negative_number = 0
            self.noisy_negative_index = []

            for i in range(self.args.warmup_sample_number, len(self.data_train)):
                if self.data_train.samples[i][3] == 1:
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



class PRODeepSyn_dataloader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.type = self.dataset.type

        self.data_train = self.dataset.data_train
        self.loader_train = FastTensorDataLoader(*self.data_train.tensor_samples(), batch_size=self.args.batch_size, shuffle=True)
        self.loader_train_evaluate = FastTensorDataLoader(*self.data_train.tensor_samples(), batch_size=self.args.batch_size, shuffle=False)
        

        if self.type == "warmup":
            if self.args.evaluate_valid:
                self.data_valid = self.dataset.data_valid
                self.loader_valid = FastTensorDataLoader(*self.data_valid.tensor_samples(), batch_size=self.args.batch_size, shuffle=False)
                
            self.data_test = self.dataset.data_test
            self.loader_test = FastTensorDataLoader(*self.data_test.tensor_samples(), batch_size=self.args.batch_size, shuffle=False)

            

    def run(self, pred=[], probalbility=[]):

        self.dataset.run(pred=pred, probability=probalbility)
        self.data_train_labeled = self.dataset.data_train_labeled
        self.probalbility_dataset = self.dataset.probalbility_dataset

        self.dataloader_train_labeled = FastTensorDataLoader(*self.dataset.data_train.tensor_samples(self.dataset.pred_idx), batch_size=self.args.batch_size, shuffle=False)

        self.probability_dataloader = DataLoader(
            self.probalbility_dataset,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )




class FastTensorDataLoader:


    def __init__(self, *tensors, batch_size=32, shuffle=False):

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class EmbDataset(Dataset):

    def __init__(self, args, drug2id_file, cell2id_file, synergy_score_file, use_folds):
        
        self.args = args
        
        
        self.drug2id = read_map(drug2id_file)
        self.cell2id = read_map(cell2id_file)
        self.samples = []
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, label, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], int(label)]
                        self.samples.append(sample)
                        sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], int(label)]
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        drug1_id, drug2_id, cell_id, label = self.samples[item]
        drug1_feat = torch.FloatTensor([drug1_id])
        drug2_feat = torch.FloatTensor([drug2_id])
        cell_feat = torch.FloatTensor([cell_id])
        label = torch.FloatTensor([int(label)])
        return drug1_feat, drug2_feat, cell_feat, label


class PPIDataset(Dataset):

    def __init__(self, exp_file):
        self.expression = np.load(exp_file)

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, item):
        return torch.LongTensor([item]), torch.FloatTensor(self.expression[item])


class AEDataset(Dataset):

    def __init__(self, feat_file):
        self.feat = np.load(feat_file)

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, item):
        return torch.FloatTensor(self.feat[item]), torch.FloatTensor(self.feat[item])


class SynergyDataset(Dataset):

    def __init__(self, drug2id_file, cell2id_file, drug_feat_file, cell_feat_file, synergy_score_file, use_folds,
                 train=True):
        self.drug2id = read_map(drug2id_file)
        self.cell2id = read_map(cell2id_file)
        self.drug_feat = np.load(drug_feat_file)
        self.cell_feat = np.load(cell_feat_file)
        self.samples = []
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, label, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], int(label)]
                        self.samples.append(sample)
                        if train:
                            sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], int(label)]
                            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        drug1_id, drug2_id, cell_id, label = self.samples[item]
        drug1_feat = torch.from_numpy(self.drug_feat[drug1_id]).float()
        drug2_feat = torch.from_numpy(self.drug_feat[drug2_id]).float()
        cell_feat = torch.from_numpy(self.cell_feat[cell_id]).float()
        label = torch.FloatTensor([int(label)])
        return drug1_feat, drug2_feat, cell_feat, label

    def drug_feat_len(self):
        return self.drug_feat.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]


class FastSynergyDataset(Dataset):

    def __init__(self, args, type, use_folds, train=True):
        
        self.args = args
        self.type = type
        
        self.drug2id = read_map(os.path.join(self.args.data_dir, 'drug', 'data_ours', 'drug2id.tsv'))
        self.cell2id = read_map(os.path.join(self.args.data_dir, 'cell', 'data_ours', 'cell2id.tsv'))
        self.drug_feat = np.load(os.path.join(self.args.data_dir, 'drug', 'data_ours', 'drug_feat.npy'))
        self.cell_feat = np.load(os.path.join(self.args.data_dir, 'cell', 'data_ours', 'cell_feat.npy'))
        
        if self.type == "warmup":
            self.synergy_score_file = os.path.join(self.args.warmup_data_dir, f'synergy_fold{str(self.args.data_partition_idx)}.tsv')
        elif self.type == "noisylabel and warmup":
            self.synergy_score_file = os.path.join(self.args.noisylabel_warmup_data_dir, f'synergy_fold{str(self.args.data_partition_idx)}.tsv')
        
        
        self.samples = []
        self.raw_samples = []
        self.train = train
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        # pdb.set_trace()
        with open(self.synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, label, fold = line.rstrip().split('\t')
                # pdb.set_trace()
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        sample = [
                            torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                            torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                            torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                            torch.FloatTensor([int(label)]),
                        ]
                        self.samples.append(sample)
                        raw_sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], int(label)]
                        self.raw_samples.append(raw_sample)
                        if train:
                            sample = [
                                torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                                torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                                torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                                torch.FloatTensor([int(label)]),
                            ]
                            self.samples.append(sample)
                            raw_sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], int(label)]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat_len(self):
        return self.drug_feat.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        return d1, d2, c, y



class DSDataset(Dataset):

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        return torch.FloatTensor(self.samples[item]), torch.FloatTensor([self.labels[item]])


def read_map(map_file):
    d = {}
    with open(map_file, 'r') as f:
        f.readline()
        for line in f:
            k, v = line.rstrip().split('\t')
            d[k] = int(v)
    return d
