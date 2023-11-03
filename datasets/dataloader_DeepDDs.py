from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import json
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader as torch_geometric_DataLoader


class DeepDDs_dataset(Dataset):
    def __init__(self, args, type=""):
        self.args = args
        self.type = type

        self.load_DeepDDs()

    def load_DeepDDs(self):
        if self.type == "warmup":
            self.data_dir = self.args.warmup_data_dir
        elif self.type == "noisylabel and warmup":
            self.data_dir = self.args.noisylabel_warmup_data_dir


        self.datafile_train = f"train_fold{str(self.args.data_partition_idx)}"
        self.datafile_test = f"test_fold{str(self.args.data_partition_idx)}"
        self.datafile_valid = f"valid_fold{str(self.args.data_partition_idx)}"

        if self.type == "warmup":
            if self.args.evaluate_valid:
                self.drug1_data_valid = TestbedDataset(
                    args=self.args,
                    root=self.data_dir, dataset=self.datafile_valid + "_drug1"
                )
                self.drug2_data_valid = TestbedDataset(
                    args=self.args,
                    root=self.data_dir, dataset=self.datafile_valid + "_drug2"
                )

            self.drug1_data_test = TestbedDataset(
                args=self.args,
                root=self.data_dir, dataset=self.datafile_test + "_drug1"
            )
            self.drug2_data_test = TestbedDataset(
                args=self.args,
                root=self.data_dir, dataset=self.datafile_test + "_drug2"
            )

        self.drug1_data_train = TestbedDataset(
            args=self.args,
            root=self.data_dir, dataset=self.datafile_train + "_drug1"
        )
        self.drug2_data_train = TestbedDataset(
            args=self.args,
            root=self.data_dir, dataset=self.datafile_train + "_drug2"
        )
        

        self.get_positive_negative_number()




    def run(self, pred=[], probability=[]):
        # train，随机
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

            if self.args.co_refinement:
                # self.drug1_data_train_labeled = TestbedDataset(
                #     root=self.data_dir,
                #     dataset=self.datafile_train + "_drug1",
                #     # probability=probability,
                # )
                # self.drug1_data_train_labeled = torch.utils.data.Subset(
                #     self.drug1_data_train_labeled, pred_idx
                # )
                # self.drug2_data_train_labeled = TestbedDataset(
                #     root=self.data_dir,
                #     dataset=self.datafile_train + "_drug2",
                #     # probability=probability,
                # )
                # self.drug2_data_train_labeled = torch.utils.data.Subset(
                #     self.drug2_data_train_labeled, pred_idx
                # )

                tensor_data = torch.tensor(probability)
                self.probalbility_dataset = TensorDataset(tensor_data)
                self.probalbility_dataset = torch.utils.data.Subset(
                    self.probalbility_dataset, pred_idx
                )

            self.drug1_data_train_labeled = torch.utils.data.Subset(
                self.drug1_data_train, pred_idx
            )
            self.drug2_data_train_labeled = torch.utils.data.Subset(
                self.drug2_data_train, pred_idx
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
        # elif mode == "unlabeled":
        #     pred_idx = (1-pred).nonzero()[0]
        #     self.drug1_data_train_unlabeled = torch.utils.data.Subset(self.drug1_data_train, pred_idx)
        #     self.drug2_data_train_unlabeled = torch.utils.data.Subset(self.drug2_data_train, pred_idx)
        # return self.drug1_data_train_labeled, self.drug2_data_train_labeled, self.drug1_data_train_unlabeled, self.drug2_data_train_unlabeled
        # return self.drug1_data_train_labeled, self.drug2_data_train_labeled

    def get_positive_negative_number(self):
        self.positive_number = 0
        self.positive_index = []
        self.negative_number = 0
        self.negative_index = []
        for i in range(len(self.drug1_data_train)):
            if self.drug1_data_train.data.y[i] == 1:
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
                if self.drug1_data_train.data.y[i] == 1:
                    self.clean_positive_number += 1
                    self.clean_positive_index.append(i)
                else:
                    self.clean_negative_number += 1
                    self.clean_negative_index.append(i)

            self.noisy_positive_number = 0
            self.noisy_positive_index = []
            self.noisy_negative_number = 0
            self.noisy_negative_index = []

            for i in range(self.args.warmup_sample_number, len(self.drug1_data_train)):
                if self.drug1_data_train.data.y[i] == 1:
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


class DeepDDs_dataloader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.type = self.dataset.type

        self.drug1_data_train = self.dataset.drug1_data_train
        self.drug2_data_train = self.dataset.drug2_data_train

        self.drug1_loader_train_evaluate = torch_geometric_DataLoader(
            self.drug1_data_train,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )
        self.drug2_loader_train_evaluate = torch_geometric_DataLoader(
            self.drug2_data_train,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )
        self.drug1_loader_train = torch_geometric_DataLoader(
            self.drug1_data_train,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )
        self.drug2_loader_train = torch_geometric_DataLoader(
            self.drug2_data_train,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )

        if self.type == "warmup":
            if self.args.evaluate_valid:
                self.drug1_data_valid = self.dataset.drug1_data_valid
                self.drug2_data_valid = self.dataset.drug2_data_valid
                self.drug1_loader_valid = torch_geometric_DataLoader(
                    self.drug1_data_valid,
                    batch_size=self.args.batch_size,
                    shuffle=None,
                    num_workers=4,
                )
                self.drug2_loader_valid = torch_geometric_DataLoader(
                    self.drug2_data_valid,
                    batch_size=self.args.batch_size,
                    shuffle=None,
                    num_workers=4,
                )

            self.drug1_data_test = self.dataset.drug1_data_test
            self.drug2_data_test = self.dataset.drug2_data_test
            self.drug1_loader_test = torch_geometric_DataLoader(
                self.drug1_data_test,
                batch_size=self.args.batch_size,
                shuffle=None,
                num_workers=4,
            )
            self.drug2_loader_test = torch_geometric_DataLoader(
                self.drug2_data_test,
                batch_size=self.args.batch_size,
                shuffle=None,
                num_workers=4,
            )
            

    def run(self, pred=[], probalbility=[]):
        # train，随机
        self.dataset.run(pred=pred, probability=probalbility)
        self.drug1_data_train_labeled = self.dataset.drug1_data_train_labeled
        self.drug2_data_train_labeled = self.dataset.drug2_data_train_labeled
        self.probalbility_dataset = self.dataset.probalbility_dataset

        self.drug1_loader_train_labeled = torch_geometric_DataLoader(
            self.drug1_data_train_labeled,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )
        self.drug2_loader_train_labeled = torch_geometric_DataLoader(
            self.drug2_data_train_labeled,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )

        self.probability_dataloader = DataLoader(
            self.probalbility_dataset,
            batch_size=self.args.batch_size,
            shuffle=None,
            num_workers=4,
        )
        # drug1_data_train_unlabeled = self.dataset.run(mode='unlabeled', pred=pred, probability=probalbility)
        # drug2_data_train_unlabeled = self.dataset.run(mode='unlabeled', pred=pred, probability=probalbility)
        # drug1_loader_train_unlabeled = torch_geometric_DataLoader(drug1_data_train_unlabeled, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        # drug2_loader_train_unlabeled = torch_geometric_DataLoader(drug2_data_train_unlabeled, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        # return drug1_data_train_labeled, drug2_data_train_labeled, drug1_loader_train_labeled, drug2_loader_train_labeled, drug1_data_train_unlabeled, drug2_data_train_unlabeled, drug1_loader_train_unlabeled, drug2_loader_train_unlabeled, drug1_data_train_labeled.data_state
        # return (
        #     drug1_data_train_labeled,
        #     drug2_data_train_labeled,
        #     drug1_loader_train_labeled,
        #     drug2_loader_train_labeled,
        # )


class TestbedDataset(InMemoryDataset):
    def __init__(
        self,
        args,
        root="/tmp",
        dataset="_drug1",
        xd=None,
        xt=None,
        y=None,
        xt_featrue=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
    ):
        
        self.args = args
        
        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print(
                "Pre-processed data found: {}, loading ...".format(
                    self.processed_paths[0]
                )
            )
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(
                "Pre-processed data {} not found, doing pre-processing...".format(
                    self.processed_paths[0]
                )
            )
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert len(xd) == len(xt) and len(xt) == len(
            y
        ), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print("number of data", data_len)
        for i in tqdm(range(data_len)):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.Tensor([labels]),
            )
            cell = self.get_cell_feature(target, xt_featrue)

            if cell == False: 
                print("cell", cell)
                sys.exit()

            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))
            GCNData.cell = torch.FloatTensor([new_cell])
            GCNData.__setitem__("c_size", torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print("Graph construction done. Saving to file.")
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def save_AUCs(AUCs, filename):
    with open(filename, "a") as f:
        f.write("\t".join(map(str, AUCs)) + "\n")


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci
