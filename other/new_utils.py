import logging
import os
import torch
import random
import numpy as np
from transformers import T5ForConditionalGeneration
from tqdm import tqdm
import math


class PrefixTree:
    def __init__(self):
        self.root = dict()

    def set(self, path):
        pointer = self.root
        for i in path:
            if i not in pointer:
                pointer[i] = dict()
            pointer = pointer[i]

    def set_all(self, path_list):
        for path in tqdm(path_list):
            self.set(path)

    def find(self, path):
        if isinstance(path, torch.Tensor):
            path = path.cpu().tolist()
        pointer = self.root
        for i in path:
            if i not in pointer:
                return []
            pointer = pointer[i]
        return list(pointer.keys())

    def __call__(self, batch_id, path):
        return self.find(path)


def get_lastpos_state(output,mask_data):
    gather_index = mask_data.sum(dim=1).unsqueeze(1) - 1
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]).long()
    output_tensor = output.gather(dim=1, index=gather_index) 

    return output_tensor.squeeze(1)


def get_logger(log_path = None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s  --%(levelname)s --%(message)s')
    cmd_handler = logging.StreamHandler()
    cmd_handler.setLevel(logging.INFO)
    cmd_handler.setFormatter(formatter)

    logger.addHandler(cmd_handler)

    if log_path is not None:
        if not os.path.exists(log_path):
            os.makedirs(log_path) 

        log_file = os.path.join(log_path,'log.out')
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_load_ckpt(model,checkpoint):
    t5model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    t5model_state_dict = t5model.state_dict()
    model_state_dict = model.state_dict()

    for name,param in t5model_state_dict.items():
        if name not in model_state_dict:
            print('model missing name:',name)
        else:
            try:
                model_state_dict[name].copy_(param)
            except:
                print(name)
                print(model_state_dict[name].size())
                print(param.size())
                print('wrong size:', name)


    return model

def save_checkpoint(save_path,model,accelerator):
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    model.save_pretrained(save_path, save_function=accelerator.save)


def compute_bpr_loss(bpr,user_hidden_state,pos_hidden_state,neg_hidden_state,
                     mean=True):
    shape = user_hidden_state.shape
    assert pos_hidden_state.shape == shape
    assert neg_hidden_state.shape == shape

    user_hidden_state = user_hidden_state.unsqueeze(dim=1) # batch,1,embed_size
    pos_hidden_state = pos_hidden_state.unsqueeze(dim=2) # batch embed_size,1
    neg_hidden_state = neg_hidden_state.unsqueeze(dim=2) # batch


    pos_score = torch.bmm(user_hidden_state,pos_hidden_state).view(shape[0])
    neg_score = torch.bmm(user_hidden_state,neg_hidden_state).view(shape[0])

    return bpr(pos_score,neg_score,mean)

def get_generate_acc(logits,labels):
    preds = torch.argmax(logits, -1)
    assert preds.shape == labels.shape
    correct = preds == labels
    acc = torch.sum(correct) / math.prod(labels.shape)

    return acc.item()

def uni_sampling(item_num):
    return np.random.randint(1,item_num+1)

def sampling_from_list(sample_list):

    return np.random.choice(sample_list)

def user_sampling(src,num):
    if len(src) < num:
        return np.random.choice(src, size=num, replace=True).tolist()
    else:
        return np.random.choice(src, size=num, replace=False).tolist()



