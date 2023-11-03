
import os
import torch
import collections
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import torch.utils.data as Data

import csv

import numpy as np
from torch.utils.data import Dataset, DataLoader

class GraphSynergy_dataset:
    def __init__(self, args, split="", dataset=None, type="", probability=[], targets=None):
        self.args = args
        self.type = type
        self.split = split
        self.targets = targets
        
        
        if self.type == "warmup":
            self.data_dir = os.path.join(
                self.args.warmup_data_dir,
                f"fold{str(self.args.data_partition_idx)}",
                split,
            )
        elif self.type == "noisylabel":
            self.data_dir = os.path.join(
                self.args.noisylabel_data_dir,
                f"fold{str(self.args.data_partition_idx)}",
                split,
            )
        elif self.type == "noisylabel and warmup":
            self.data_dir = os.path.join(
                self.args.noisylabel_warmup_data_dir,
                f"fold{str(self.args.data_partition_idx)}",
                split,
            )
        else:
            raise ValueError(f"type must be warmup or noisylabel, but got {type}.")

        self.aux_data_dir = self.args.aux_data_dir
        self.score, self.threshold = self.args.score.split(" ")

        self.drug_combination_df, _, _, _ = self.load_data()
        self.ppi_df = dataset.ppi_df
        self.cpi_df = dataset.cpi_df
        self.dpi_df = dataset.dpi_df

        # get node map
        self.node_map_dict = dataset.node_map_dict
        self.node_num_dict = dataset.node_num_dict

        # remap the node in the data frame
        self.df_node_remap()
        # drug combinations data remapping
        self.feature_index = self.drug_combination_process()

        # create dataset
        self.dataset = self.create_dataset(probability=probability)

    def load_data(self):
        drug_combination_df = pd.read_csv(
            os.path.join(self.data_dir, "drug_combinations.csv")
        )
        ppi_df = pd.read_excel(
            os.path.join(self.aux_data_dir, "protein-protein_network.xlsx")
        )
        cpi_df = pd.read_csv(os.path.join(self.aux_data_dir, "cell_protein.csv"))
        dpi_df = pd.read_csv(os.path.join(self.aux_data_dir, "drug_protein.csv"))

        return drug_combination_df, ppi_df, cpi_df, dpi_df

    # def get_node_map_dict(self):
    #     protein_node = list(set(self.ppi_df['protein_a']) | set(self.ppi_df['protein_b']))
    #     cell_node = list(set(self.cpi_df['cell']))
    #     drug_node = list(set(self.dpi_df['drug']))

    #     node_num_dict = {'protein': len(protein_node), 'cell': len(cell_node), 'drug': len(drug_node)}

    #     mapping = {protein_node[idx]:idx for idx in range(len(protein_node))}
    #     mapping.update({cell_node[idx]:idx for idx in range(len(cell_node))})
    #     mapping.update({drug_node[idx]:idx for idx in range(len(drug_node))})

    #     # display data info
    #     print('undirected graph')
    #     print('# proteins: {0}, # drugs: {1}, # cells: {2}'.format(
    #             len(protein_node), len(drug_node), len(cell_node)))
    #     print('# protein-protein interactions: {0}, # drug-protein associations: {1}, # cell-protein associations: {2}'.format(
    #         len(self.ppi_df), len(self.dpi_df), len(self.cpi_df)))

    #     return mapping, node_num_dict

    def df_node_remap(self):
        # self.ppi_df['protein_a'] = self.ppi_df['protein_a'].map(self.node_map_dict)
        # self.ppi_df['protein_b'] = self.ppi_df['protein_b'].map(self.node_map_dict)
        # self.ppi_df = self.ppi_df[['protein_a', 'protein_b']]

        # self.cpi_df['cell'] = self.cpi_df['cell'].map(self.node_map_dict)
        # self.cpi_df['protein'] = self.cpi_df['protein'].map(self.node_map_dict)
        # self.cpi_df = self.cpi_df[['cell', 'protein']]

        # self.dpi_df['drug'] = self.dpi_df['drug'].map(self.node_map_dict)
        # self.dpi_df['protein'] = self.dpi_df['protein'].map(self.node_map_dict)
        # self.dpi_df = self.dpi_df[['drug', 'protein']]

        self.drug_combination_df["drug1_db"] = self.drug_combination_df["drug1_db"].map(
            self.node_map_dict
        )
        self.drug_combination_df["drug2_db"] = self.drug_combination_df["drug2_db"].map(
            self.node_map_dict
        )
        self.drug_combination_df["cell"] = self.drug_combination_df["cell"].map(
            self.node_map_dict
        )

    def drug_combination_process(self):
        self.drug_combination_df["synergistic"] = [0] * len(self.drug_combination_df)
        self.drug_combination_df.loc[
            self.drug_combination_df[self.score] > eval(self.threshold), "synergistic"
        ] = 1
        self.drug_combination_df.to_csv(
            os.path.join(self.data_dir, "drug_combination_processed.csv"), index=False
        )

        self.drug_combination_df = self.drug_combination_df[
            ["cell", "drug1_db", "drug2_db", "synergistic"]
        ]

        return {"cell": 0, "drug1": 1, "drug2": 2}

    def create_dataset(self, probability=[]):
        # shape [n_data, 3]
        feature = torch.from_numpy(self.drug_combination_df.to_numpy())

        if self.targets != None:
            label = self.targets
        else:
            # shape [n_data, 1]
            label = torch.from_numpy(self.drug_combination_df[["synergistic"]].to_numpy())




        self.label = label


        # change tensor type
        feature = feature.type(torch.LongTensor)
        label = label.type(torch.FloatTensor)
        # create dataset
        # dataset = Data.TensorDataset(feature, label)
        
        
        if self.args.noisy_data_ratio != 2:
            
            
            if self.args.part_shuffle == True and self.split == "train":
                
                print("Start selecting")
                print(f"warmup_sample_number {str(self.args.warmup_sample_number)}")
                print(f"selected data {str(self.args.noisy_data_ratio * (len(feature) - self.args.warmup_sample_number))}")
                print(f"all data {str(len(feature))}")

                # 选择要shuffle的部分
                sub_feature = feature[self.args.warmup_sample_number:]
                sub_label = label[self.args.warmup_sample_number:]

                # 使用torch.randperm函数来生成随机的索引，然后根据这些索引重新排列数据
                shuffled_sub_feature = sub_feature[torch.randperm(sub_feature.size(0))]
                shuffled_sub_label = sub_label[torch.randperm(sub_feature.size(0))]

                # 将重新排列后的部分放回原始数据中的相应位置
                feature[self.args.warmup_sample_number:, :] = shuffled_sub_feature
                label[self.args.warmup_sample_number:, :] = shuffled_sub_label

            
            feature = feature[:int(self.args.warmup_sample_number + self.args.noisy_data_ratio * (len(feature) - self.args.warmup_sample_number)), :]
            label = label[:int(self.args.warmup_sample_number + self.args.noisy_data_ratio * (len(label) - self.args.warmup_sample_number)), :]
        
        

        if probability != []:
            probalbility = torch.from_numpy(np.array(probability))
            probalbility = probalbility.type(torch.FloatTensor)
            dataset = Data.TensorDataset(feature, label, probalbility)
        else:
            dataset = Data.TensorDataset(feature, label)

        return dataset


class GraphSynergy_dataset_info:
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.aux_data_dir = self.args.aux_data_dir
        self.score, self.threshold = self.args.score.split(" ")
        self.n_hop = self.args.n_hop
        self.n_memory = self.args.n_memory

        self.data_dir = os.path.join(
            self.args.data_dir, f"fold{str(self.args.data_partition_idx)}", "train"
        )

        self.batch_size = self.args.batch_size

        self.num_workers = 1

        # load data
        (
            self.drug_combination_df,
            self.ppi_df,
            self.cpi_df,
            self.dpi_df,
        ) = self.load_data()
        # get node map
        self.node_map_dict, self.node_num_dict = self.get_node_map_dict()
        # remap the node in the data frame
        self.df_node_remap()
        # drug combinations data remapping
        self.feature_index = self.drug_combination_process()

        # create dataset
        # self.dataset = self.create_dataset()

        # build the graph
        self.graph = self.build_graph()
        # get target dict
        self.cell_protein_dict, self.drug_protein_dict = self.get_target_dict()
        # some indexs
        self.cells = list(self.cell_protein_dict.keys())
        self.drugs = list(self.drug_protein_dict.keys())
        # get neighbor set
        self.cell_neighbor_set = self.get_neighbor_set(
            items=self.cells, item_target_dict=self.cell_protein_dict
        )
        self.drug_neighbor_set = self.get_neighbor_set(
            items=self.drugs, item_target_dict=self.drug_protein_dict
        )
        # save data
        self._save()

    def get_cell_neighbor_set(self):
        return self.cell_neighbor_set

    def get_drug_neighbor_set(self):
        return self.drug_neighbor_set

    def get_feature_index(self):
        return self.feature_index

    def get_node_num_dict(self):
        return self.node_num_dict

    def load_data(self):
        drug_combination_df = pd.read_csv(
            os.path.join(self.data_dir, "drug_combinations.csv")
        )
        ppi_df = pd.read_excel(
            os.path.join(self.aux_data_dir, "protein-protein_network.xlsx")
        )
        cpi_df = pd.read_csv(os.path.join(self.aux_data_dir, "cell_protein.csv"))
        dpi_df = pd.read_csv(os.path.join(self.aux_data_dir, "drug_protein.csv"))

        return drug_combination_df, ppi_df, cpi_df, dpi_df

    def get_node_map_dict(self):
        protein_node = list(
            set(self.ppi_df["protein_a"]) | set(self.ppi_df["protein_b"])
        )
        cell_node = list(set(self.cpi_df["cell"]))
        drug_node = list(set(self.dpi_df["drug"]))

        node_num_dict = {
            "protein": len(protein_node),
            "cell": len(cell_node),
            "drug": len(drug_node),
        }

        mapping = {protein_node[idx]: idx for idx in range(len(protein_node))}
        mapping.update({cell_node[idx]: idx for idx in range(len(cell_node))})
        mapping.update({drug_node[idx]: idx for idx in range(len(drug_node))})

        # display data info
        print("undirected graph")
        print(
            "# proteins: {0}, # drugs: {1}, # cells: {2}".format(
                len(protein_node), len(drug_node), len(cell_node)
            )
        )
        print(
            "# protein-protein interactions: {0}, # drug-protein associations: {1}, # cell-protein associations: {2}".format(
                len(self.ppi_df), len(self.dpi_df), len(self.cpi_df)
            )
        )

        return mapping, node_num_dict

    def df_node_remap(self):
        self.ppi_df["protein_a"] = self.ppi_df["protein_a"].map(self.node_map_dict)
        self.ppi_df["protein_b"] = self.ppi_df["protein_b"].map(self.node_map_dict)
        self.ppi_df = self.ppi_df[["protein_a", "protein_b"]]

        self.cpi_df["cell"] = self.cpi_df["cell"].map(self.node_map_dict)
        self.cpi_df["protein"] = self.cpi_df["protein"].map(self.node_map_dict)
        self.cpi_df = self.cpi_df[["cell", "protein"]]

        self.dpi_df["drug"] = self.dpi_df["drug"].map(self.node_map_dict)
        self.dpi_df["protein"] = self.dpi_df["protein"].map(self.node_map_dict)
        self.dpi_df = self.dpi_df[["drug", "protein"]]

        self.drug_combination_df["drug1_db"] = self.drug_combination_df["drug1_db"].map(
            self.node_map_dict
        )
        self.drug_combination_df["drug2_db"] = self.drug_combination_df["drug2_db"].map(
            self.node_map_dict
        )
        self.drug_combination_df["cell"] = self.drug_combination_df["cell"].map(
            self.node_map_dict
        )

    def drug_combination_process(self):
        self.drug_combination_df["synergistic"] = [0] * len(self.drug_combination_df)
        self.drug_combination_df.loc[
            self.drug_combination_df[self.score] > eval(self.threshold), "synergistic"
        ] = 1
        self.drug_combination_df.to_csv(
            os.path.join(self.data_dir, "drug_combination_processed.csv"), index=False
        )

        self.drug_combination_df = self.drug_combination_df[
            ["cell", "drug1_db", "drug2_db", "synergistic"]
        ]

        return {"cell": 0, "drug1": 1, "drug2": 2}

    def build_graph(self):
        tuples = [tuple(x) for x in self.ppi_df.values]
        graph = nx.Graph()
        graph.add_edges_from(tuples)
        return graph

    def get_target_dict(self):
        cp_dict = collections.defaultdict(list)
        cell_list = list(set(self.cpi_df["cell"]))
        for cell in cell_list:
            cell_df = self.cpi_df[self.cpi_df["cell"] == cell]
            target = list(set(cell_df["protein"]))
            cp_dict[cell] = target

        dp_dict = collections.defaultdict(list)
        drug_list = list(set(self.dpi_df["drug"]))
        for drug in drug_list:
            drug_df = self.dpi_df[self.dpi_df["drug"] == drug]
            target = list(set(drug_df["protein"]))
            dp_dict[drug] = target

        return cp_dict, dp_dict

    def create_dataset(self):
        # shuffle data
        if self.shuffle:
            self.drug_combination_df = self.drug_combination_df.sample(
                frac=1, random_state=1
            )
        # shape [n_data, 3]
        feature = torch.from_numpy(self.drug_combination_df.to_numpy())
        # shape [n_data, 1]
        label = torch.from_numpy(self.drug_combination_df[["synergistic"]].to_numpy())
        # change tensor type
        feature = feature.type(torch.LongTensor)
        label = label.type(torch.FloatTensor)
        
        # create dataset
        dataset = Data.TensorDataset(feature, label)
        return dataset

    def get_neighbor_set(self, items, item_target_dict):
        print("constructing neighbor set ...")

        neighbor_set = collections.defaultdict(list)
        for item in items:
            for hop in range(self.n_hop):
                # use the target directly
                if hop == 0:
                    replace = len(item_target_dict[item]) < self.n_memory
                    target_list = list(
                        np.random.choice(
                            item_target_dict[item], size=self.n_memory, replace=replace
                        )
                    )
                else:
                    # use the last one to find k+1 hop neighbors
                    origin_nodes = neighbor_set[item][-1]
                    neighbors = []
                    for node in origin_nodes:
                        neighbors += self.graph.neighbors(node)
                    # sample
                    replace = len(neighbors) < self.n_memory
                    target_list = list(
                        np.random.choice(neighbors, size=self.n_memory, replace=replace)
                    )

                neighbor_set[item].append(target_list)

        return neighbor_set

    def _save(self):
        with open(os.path.join(self.data_dir, "node_map_dict.pickle"), "wb") as f:
            pickle.dump(self.node_map_dict, f)
        with open(os.path.join(self.data_dir, "cell_neighbor_set.pickle"), "wb") as f:
            pickle.dump(self.cell_neighbor_set, f)
        with open(os.path.join(self.data_dir, "drug_neighbor_set.pickle"), "wb") as f:
            pickle.dump(self.drug_neighbor_set, f)


class GraphSynergy_dataloader:
    def __init__(self, args, dataset, type="", valid=False, test=False):
        self.args = args
        self.type = type
        self.dataset = dataset

        self.training_GraphSynergy_dataset = GraphSynergy_dataset(
            self.args, split="train", dataset=self.dataset, type=type
        )
        self.train_dataset = self.training_GraphSynergy_dataset.dataset


        self.train_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=1,
        )
        self.evaluate_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=1,
        )

        if valid == True:
            self.valid_dataset = GraphSynergy_dataset(
                self.args, split="valid", dataset=self.dataset, type=type
            ).dataset
            self.valid_data_loader = DataLoader(
                dataset=self.valid_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=1,
            )
        if test == True:
            self.test_dataset = GraphSynergy_dataset(
                self.args, split="test", dataset=self.dataset, type=type
            ).dataset
            self.test_data_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=1,
            )

        self.get_positive_negative_number(self.train_dataset)

    def run(self, pred=[], probability=[], targets=None, drop=True):
        # train，随机
        pred_clean = pred[:self.args.warmup_sample_number]
        pred_noisy = pred[self.args.warmup_sample_number:]

        pred_idx_clean_noisy = pred.nonzero()[0]
        pred_idx_clean = pred_clean.nonzero()[0]
        pred_idx_noisy = pred_noisy.nonzero()[0]
        pred_idx_noisy += self.args.warmup_sample_number

        label = self.training_GraphSynergy_dataset.label
        self.output_data = [probability, label]

        if not self.args.noisylabel_only_noisy:
            pred_idx = pred_idx_clean_noisy
        else:
            pred_idx_clean_ones = np.arange(0, (len(pred_clean)), 1, dtype=int)
            pred_idx = np.concatenate((pred_idx_clean_ones, pred_idx_noisy))

        # if self.args.co_refinement:
        self.train_dataset_labeled = GraphSynergy_dataset(
            self.args,
            split="train",
            dataset=self.dataset,
            type=self.type,
            probability=probability,
            targets=targets,
        ).dataset

        if drop == True:
            self.train_dataset_labeled = torch.utils.data.Subset(
                self.train_dataset_labeled, pred_idx
            )

        self.data_loader_labeled = DataLoader(
            dataset=self.train_dataset_labeled,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=1,
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

        return self.train_dataset_labeled, self.data_loader_labeled

    def get_positive_negative_number(self, dataset):
        self.positive_number = 0
        self.positive_index = []
        self.negative_number = 0
        self.negative_index = []

        for i in range(len(dataset)):
            if dataset.tensors[1][i] == 1:
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
                if dataset.tensors[1][i] == 1:
                    self.clean_positive_number += 1
                    self.clean_positive_index.append(i)
                else:
                    self.clean_negative_number += 1
                    self.clean_negative_index.append(i)

            self.noisy_positive_number = 0
            self.noisy_positive_index = []
            self.noisy_negative_number = 0
            self.noisy_negative_index = []

            for i in range(self.args.warmup_sample_number, len(dataset)):
                if dataset.tensors[1][i] == 1:
                    self.noisy_positive_number += 1
                    self.noisy_positive_index.append(i)
                else:
                    self.noisy_negative_number += 1
                    self.noisy_negative_index.append(i)

            self.data_state["clean_positive_number"] = self.clean_positive_number,
            self.data_state["clean_positive_index"] = self.clean_positive_index,
            self.data_state["clean_negative_number"] = self.clean_negative_number,
            self.data_state["clean_negative_index"] = self.clean_negative_index,
            self.data_state["noisy_positive_number"] = self.noisy_positive_number,
            self.data_state["noisy_positive_index"] = self.noisy_positive_index,
            self.data_state["noisy_negative_number"] = self.noisy_negative_number,
            self.data_state["noisy_negative_index"] = self.noisy_negative_index,