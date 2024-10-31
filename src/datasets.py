import sys
import pandas as pd
sys.path.append('../')
import torch
from torch.utils.data import Dataset,DataLoader
import json
import os
import random
from tqdm import tqdm
from src.data_process_utils import trans_cid2int_list,get_inter_list,get_user_interlist,get_item_interlist
from other.config_class import DataFileConfig,DataProcessConfig
from other.new_utils import uni_sampling,user_sampling,sampling_from_list

class BasedTrainDataset(Dataset):
    def __init__(self,
                 inter_list, user_seq, item_corpus,
                 process_config,
                 tokenizer):
        super(BasedTrainDataset, self).__init__()
        self.dataset: list = inter_list
        self.item_corpus: dict = item_corpus
        self.user_seq: dict = user_seq #
        self.tokenizer = tokenizer
        self.pro_config = process_config


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        assert len(self.dataset[idx]) == 2,'the dataset must consist of (user,item) interation pairs.'
        uid, iid = self.dataset[idx]

        user_atomid_list = self.user_seq[uid][:]
        pos_item_content = [self.item_corpus[str(iid)].get('content')]
        pos_item_atomid = [int(self.item_corpus[str(iid)].get('atom_id'))]
        label_item = self.item_corpus[str(iid)].get('cid')

        item_num = len(self.item_corpus.keys())
        nid = uni_sampling(item_num)
        while nid in user_atomid_list:
            nid = uni_sampling(item_num)

        neg_item_content = [self.item_corpus[str(nid)].get('content')]
        neg_item_atomid = [int(self.item_corpus[str(nid)].get('atom_id'))]
        neg_label_item = self.item_corpus[str(nid)].get('cid')

        while user_atomid_list.count(iid) != 0:
            user_atomid_list.remove(iid)
        if len(user_atomid_list) == 0:
            print("!!!!!")

        sample_num = min(self.pro_config.sample_item_num,len(user_atomid_list))
        user_atomid_list = random.sample(user_atomid_list,sample_num)

        user_content = [self.item_corpus[str(i_id)].get('content') for i_id in user_atomid_list]

        return {
            'item_content': pos_item_content,
            'item_atomid': pos_item_atomid,
            'neg_item_content': neg_item_content,
            'neg_item_atomid': neg_item_atomid,
            'user_content': user_content,
            'user_atomid_list': user_atomid_list,
            'neg_label_item': neg_label_item,
            'label_item': label_item
        }

    def collate_fn(self, batch_data):
        pos_input_dict = self.tokenizer([data['item_content'] for data in batch_data],return_tensor=True)
        pos_input_dict['atom_input_ids'] = torch.LongTensor([data['item_atomid'] for data in batch_data])
        labels = {'labels': torch.LongTensor([data['label_item'] for data in batch_data])}

        neg_input_dict = self.tokenizer([data['neg_item_content'] for data in batch_data], return_tensor=True)
        neg_input_dict['atom_input_ids'] = torch.LongTensor([data['neg_item_atomid'] for data in batch_data])
        neg_labels = {'labels': torch.LongTensor([data['neg_label_item'] for data in batch_data])}

        user_input_dict = self.tokenizer([data['user_content'] for data in batch_data],return_tensor=True)
        user_atomlist = [data['user_atomid_list'] for data in batch_data]
        user_atomlist = self.user_atomid_cutting_v2(user_atomlist,user_input_dict['atom_index'].tolist())
        user_atomlist = self.padding(user_atomlist,return_tensor=True)
        user_input_dict['atom_input_ids'] = user_atomlist

        return user_input_dict,pos_input_dict,neg_input_dict,labels,neg_labels

    def user_atomid_cutting_v2(self,batch_data,src):
        new_batch_data = []
        for idx,data in enumerate(batch_data):
            item_num = len(src[idx]) - src[idx].count(0)
            new_batch_data.append(data[:item_num])

        return new_batch_data

    def padding(self,batch_data,padding_idx=0,return_tensor=False):
        max_length = max([len(data) for data in batch_data])

        batch_pad_data = []
        for data in batch_data:
            length_to_pad = max_length - len(data)
            data += [padding_idx] * length_to_pad
            batch_pad_data.append(data)

        if return_tensor:
            batch_pad_data = torch.LongTensor(batch_pad_data)

        return batch_pad_data

class v4TrainDataset(BasedTrainDataset):
    def __init__(self,
                 inter_list, user_seq, item_seq,
                 item_corpus,
                 process_config,
                 tokenizer):
        super(v4TrainDataset, self).__init__(inter_list, user_seq,item_corpus,process_config,tokenizer)
        self.item_seq:dict = item_seq

    def __getitem__(self, idx):
        ans_dict = super(v4TrainDataset, self).__getitem__(idx)

        # pos抽样
        pos_id = ans_dict['item_atomid'][0]
        pos_user_atomids = user_sampling(self.item_seq[pos_id],self.pro_config.sample_user_num)
        # neg抽样
        neg_id = ans_dict['neg_item_atomid'][0]
        neg_list = self.item_seq.get(neg_id,[0])
        neg_user_atomids = user_sampling(neg_list,self.pro_config.sample_user_num)

        ans_dict.update({
            'item_user_atomids':pos_user_atomids,
            'neg_item_user_atomids':neg_user_atomids
        })

        return ans_dict

class v7CL2TrainDataset(v4TrainDataset):
    def __init__(self,
                 inter_list, user_seq, item_seq,
                 item_corpus,
                 process_config,
                 tokenizer,cid_neg_dict):
        super(v7CL2TrainDataset, self).__init__(inter_list, user_seq,item_seq,
                                                item_corpus,process_config,
                                                tokenizer)

        self.cid_neg_dict = cid_neg_dict

    def __getitem__(self, idx):
        assert len(self.dataset[idx]) == 2,'the dataset must consist of (user,item) interation pairs.'
        uid, iid = self.dataset[idx]

        user_atomid_list = self.user_seq[uid][:]
        pos_item_content = [self.item_corpus[str(iid)].get('content')]
        pos_item_atomid = [int(self.item_corpus[str(iid)].get('atom_id'))]
        label_item = self.item_corpus[str(iid)].get('cid')

        aug_set = self.get_aug_set(iid)
        if len(aug_set) == 0:
            aug_set = [iid]
        aug_iid = sampling_from_list(aug_set)
        aug_item_content = [self.item_corpus[str(aug_iid)].get('content')]
        aug_item_atomid = [int(self.item_corpus[str(aug_iid)].get('atom_id'))]

        item_num = len(self.item_corpus.keys())
        nid = uni_sampling(item_num)
        while (nid in user_atomid_list) or (nid in aug_set):
            nid = uni_sampling(item_num)

        neg_item_content = [self.item_corpus[str(nid)].get('content')]
        neg_item_atomid = [int(self.item_corpus[str(nid)].get('atom_id'))]
        neg_label_item = self.item_corpus[str(nid)].get('cid')

        while user_atomid_list.count(iid) != 0:
            user_atomid_list.remove(iid)
        if len(user_atomid_list) == 0:
            print("!!!!!")

        sample_num = min(self.pro_config.sample_item_num,len(user_atomid_list))
        user_atomid_list = random.sample(user_atomid_list,sample_num)

        user_content = [self.item_corpus[str(i_id)].get('content') for i_id in user_atomid_list]

        pos_user_atomids = user_sampling(self.item_seq[iid], self.pro_config.sample_user_num)
        aug_ulist = self.item_seq.get(aug_iid,[0])
        aug_user_atomids = user_sampling(aug_ulist,self.pro_config.sample_user_num)
        neg_list = self.item_seq.get(nid, [0])
        neg_user_atomids = user_sampling(neg_list, self.pro_config.sample_user_num)

        return {
            'item_content': pos_item_content,
            'item_atomid': pos_item_atomid,
            'aug_item_content':aug_item_content,
            'aug_item_atomid':aug_item_atomid,
            'neg_item_content': neg_item_content,
            'neg_item_atomid': neg_item_atomid,
            'user_content': user_content,
            'user_atomid_list': user_atomid_list,
            'neg_label_item': neg_label_item,
            'label_item': label_item,
            'item_user_atomids': pos_user_atomids,
            'aug_item_user_atomids':aug_user_atomids,
            'neg_item_user_atomids': neg_user_atomids
        }

    def collate_fn(self, batch_data):
        user_input_dict, pos_input_dict, neg_input_dict, labels, neg_labels = super(v7CL2TrainDataset, self).collate_fn(
            batch_data)

        aug_input_dict = self.tokenizer([data['aug_item_content'] for data in batch_data],return_tensor=True,ty='item')
        aug_input_dict['atom_input_ids'] = torch.LongTensor([data['aug_item_atomid'] for data in batch_data])
        aug_input_dict['user_atom_ids'] = torch.LongTensor([data['aug_item_user_atomids'] for data in batch_data])

        return user_input_dict, pos_input_dict, neg_input_dict, labels, neg_labels,aug_input_dict

    def get_aug_set(self,iid):
        aug_set = self.cid_neg_dict[str(iid)]['sec']

        return aug_set

def get_train_dataloader_augitem(data_config:DataFileConfig,
                         pro_config:DataProcessConfig,
                         tokenizer,
                         batch_size = 20,
                         num_workers = 4,
                        return_dataset=False):
    with open(data_config.corpus_file, 'r') as f:
        item_corpus = json.load(f)
    cid_eg = item_corpus['1']['cid']
    if not isinstance(cid_eg,list):
        item_corpus = trans_cid2int_list(item_corpus)

    inter_file = os.path.join(data_config.dataset_path, 'train_time_inter.csv')
    user_seq = get_user_interlist(inter_file)
    inter_list = get_inter_list(inter_file)
    item_seq = get_item_interlist(inter_file)

    cid_neg_file = os.path.join(data_config.dataset_path,'cid_neg_dict.json')
    with open(cid_neg_file,'r') as f:
        cid_neg_dict = json.load(f)

    dataset = v7CL2TrainDataset(inter_list, user_seq,item_seq,item_corpus,
                                pro_config, tokenizer,cid_neg_dict)
    print('dataset.class', dataset.__class__.__name__)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=dataset.collate_fn,shuffle=True,num_workers=num_workers)

    if return_dataset:
        return dataset,dataloader
    return dataloader

class BasedValidDataset(Dataset):
    def __init__(self,user_seq,item_corpus,target_seq,tokenizer,
                 mode='latest',sample_item_num=10):
        super(BasedValidDataset, self).__init__()
        self.item_corpus: dict = item_corpus
        self.user_seq: dict = user_seq
        self.dataset = list(target_seq.keys())
        user_num = max(len(user_seq.keys()),len(target_seq.keys()))
        self.total_user_num = user_num
        self.tokenizer = tokenizer
        self.target_seq:dict = target_seq
        self.mode = mode
        self.sample_item_num = sample_item_num

    def __getitem__(self,idx):
        user = self.dataset[idx]
        user_intered = self.user_seq[user]
        user_target = self.target_seq[user]

        assert self.mode in ['random','latest'],'the mode is {}, which is not in [random,latest]'.format(self.mode)
        if self.mode == 'latest':
            user_atomid_list = self.user_seq[user][::-1]

        elif self.mode == 'random':
            sample_num = min(self.sample_item_num, len(user_intered))
            user_atomid_list = random.sample(user_intered,sample_num)
        user_content = [self.item_corpus[str(i_id)].get('content') for i_id in user_atomid_list]

        return{
            'content':user_content,
            'atomid':user_atomid_list,
            'target':user_target,
            'intered':user_intered
        }


    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch_data):

        user_input_dict = self.tokenizer([data['content'] for data in batch_data], return_tensor=True)
        user_atomlist = [data['atomid'] for data in batch_data]
        user_atomlist = self.user_atomid_cutting(user_atomlist,user_input_dict['content_input_ids'].tolist())
        user_atomlist = self.padding(user_atomlist,return_tensor=True)
        user_input_dict['atom_input_ids'] = user_atomlist

        user_target = self.padding([data['target'] for data in batch_data])
        user_intered = self.padding([data['intered'] for data in batch_data])

        return user_input_dict,user_target,user_intered

    def padding(self,batch_data,padding_idx=0,return_tensor=False):
        max_length = max([len(data) for data in batch_data])

        batch_pad_data = []
        for data in batch_data:
            length_to_pad = max_length - len(data)
            data += [padding_idx] * length_to_pad
            batch_pad_data.append(data)

        if return_tensor:
            batch_pad_data = torch.LongTensor(batch_pad_data)

        return batch_pad_data


    def user_atomid_cutting(self,batch_data,src):
        new_batch_data = []
        for idx,data in enumerate(batch_data):
            item_num = src[idx].count(self.tokenizer.atom_pad_id)
            new_batch_data.append(data[:item_num])

        return new_batch_data

def get_valid_dataloader_timeorder(data_config:DataFileConfig,
                         pro_config:DataProcessConfig,
                         tokenizer,
                         batch_size = 20,
                         num_workers = 4,
                         split='valid',
                        ):
    with open(data_config.corpus_file, 'r') as f:
        item_corpus = json.load(f)
    cid_eg = item_corpus['1']['cid']
    if not isinstance(cid_eg,list):
        item_corpus = trans_cid2int_list(item_corpus)

    inter_file = os.path.join(data_config.dataset_path, 'train_time_inter.csv')
    user_seq = get_user_interlist(inter_file)

    assert split in ['valid','test'],'split is either test or valid,now is {}'.format(split)
    valid_file = os.path.join(data_config.dataset_path, '{}_inter.csv'.format(split))
    target_seq = get_user_interlist(valid_file,time_order=False)

    dataset = BasedValidDataset(user_seq,item_corpus,target_seq,tokenizer,
                              mode=pro_config.mode,sample_item_num=pro_config.sample_item_num)
    print('valid dataset type:', dataset.__class__.__name__)
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset.collate_fn,
                            num_workers=num_workers)

    return dataloader, dataset