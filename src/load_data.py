import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import params

# Torch数据集导入类
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.sentence = dataframe['sentence']
        self.label = self.data.label
        tmp_list = []
        
        for layer in self.label:
            ttmp_list = []
            tmp_str = '[' + layer.split('[')[1]
            check_list = json.loads(tmp_str)
            
            for i in range(0, params.MAX_LABEL_SIZE):
                if i in check_list:
                    ttmp_list.append(1)
                else:
                    ttmp_list.append(0)
            tmp_list.append(ttmp_list)
        self.label = tmp_list
        
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        sentence = str(self.sentence[index])
        output = json.loads(self.data["alpaca"][index])["output"]

        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation='only_second'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'label': torch.tensor(self.label[index], dtype=torch.float),
                'sentence': sentence,
                'output': output
               }

def build_graph(dataset, device, tokenizer, max_len):
    ontology = dataset["ontology"]
    graph = {"ontology": dataset["ontology"], "nodes": [], "adj": [], "node2id": {}, "inputs": {'ids': [], 'mask': [], 'token_type_ids': []}}
    # print(len(ontology))
    for triple in ontology:
        graph["node2id"][triple["rel"]] = len(graph["nodes"])
        graph["nodes"].append(triple["rel"])
    rel_num = len(graph["nodes"])
    for triple in ontology:
        if triple["sub"] not in graph['nodes']:
            graph["node2id"][triple["sub"]] = len(graph["nodes"])
            graph["nodes"].append(triple["sub"])
        if triple["obj"] not in graph['nodes']:
            graph["node2id"][triple["obj"]] = len(graph["nodes"])
            graph["nodes"].append(triple["obj"])
    ent_num = len(graph["nodes"]) - rel_num
    graph['adj'] = [[0. for _ in range(len(graph['nodes']))] for _ in range(len(graph['nodes']))]
    
    for triple in ontology:
        graph['adj'][graph["node2id"][triple["sub"]]][graph["node2id"][triple["rel"]]] = 1.
        graph['adj'][graph["node2id"][triple["obj"]]][graph["node2id"][triple["rel"]]] = 1.
    
    label_num = [0 for _ in range(rel_num)]
    for label_json in dataset["label"]:
        label = json.loads(label_json)
        for i, pos in enumerate(label):
            label_num[pos] += 1
            for j in range(i + 1, len(label)):
                graph['adj'][pos][label[j]] += 1
                graph['adj'][label[j]][pos] += 1
    
    for i in range(rel_num):
        for j in range(i + 1, rel_num):
            graph['adj'][i][j] /= label_num[i] + 1
            graph['adj'][j][i] /= label_num[j] + 1
            
    for node in graph["nodes"]:
        inputs = tokenizer.encode_plus(
            graph["nodes"],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation='only_second'
        )
    graph["adj"] = torch.tensor(graph["adj"], dtype=torch.float, device=device)
    return graph

def data_loader(dataset_path, device):
    dataset_train = json.loads(open(dataset_path + "train_dataset.json", 'r', encoding='utf-8').read())
    dataset_val = json.loads(open(dataset_path + "val_dataset.json", 'r', encoding='utf-8').read())
    df_train = {}
    df_val = {}
    df_train = pd.DataFrame({'sentence': dataset_train['sentence'], 'label': dataset_train['label'], "alpaca": dataset_train['alpaca']})
    df_val = pd.DataFrame({'sentence': dataset_val['sentence'], 'label': dataset_val['label'], "alpaca": dataset_val['alpaca']})
    train_dataset = df_train[['sentence', 'label', 'alpaca']].copy()
    valid_dataset = df_val[['sentence', 'label', 'alpaca']].copy()
    print(f"FULL Dataset: {train_dataset.shape + valid_dataset.shape}, "
          f"TRAIN Dataset: {train_dataset.shape}, "
          f"TEST Dataset: {valid_dataset.shape}")
    
    tokenizer = BertTokenizer.from_pretrained(params.BERT_MODEL_PATH)
    training_set = CustomDataset(train_dataset, tokenizer, params.MAX_LEN)
    validation_set = CustomDataset(valid_dataset, tokenizer, params.MAX_LEN)
    train_params = {'batch_size': params.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': params.VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **test_params)
    return training_loader, validation_loader, build_graph(dataset_train, device, tokenizer, 32)
