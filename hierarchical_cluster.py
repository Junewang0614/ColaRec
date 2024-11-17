import os.path
from k_means_constrained import KMeansConstrained
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import json
import torch
import time
import accelerate
# import ipdb

class Solution:
    def __init__(self,item_embed:torch.Tensor,
                 length=3,cluster=32,**kwargs):
        self.colla_embed = item_embed

        self.length = length
        self.cluster = cluster
        self.default_config ={
                'n_clusters': self.cluster,
                'max_iter': 300,
                'n_init': 10,
                'random_state': 0,
            }
        self.default_config.update(kwargs)
        self.all_df = None
        self.total_num = self.colla_embed.shape[0]-1
        self.all_dict = {key:[] for key in range(1,self.total_num+1)}


    def kmeans_cluster(self,level,item_ids):
        if level >= self.length:
            return

        now_config = self.default_config.copy()
        now_config['size_max'] = self.cluster ** (self.length - level)
        now_config['n_clusters'] = min(self.cluster,len(item_ids))
        now_config['size_max'] = min(now_config['size_max'], len(item_ids))
        kmeans = KMeansConstrained(**now_config)
        embeds = self.colla_embed[item_ids].cpu().numpy()
        kmeans.fit(embeds)

        now_labels = kmeans.labels_
        for idx,aid in enumerate(item_ids.tolist()):
            self.all_dict[aid].append(now_labels[idx]+1)


        sub_labels = np.unique(now_labels)
        for sub in sub_labels:
            indices = np.where(now_labels == sub)
            indices = indices[0].tolist()
            sub_id_list = item_ids[indices]
            self.kmeans_cluster(level+1,sub_id_list)


def get_gid_df(gid_dict):
    values_data = []
    for key,value in gid_dict.items():
        val = value + [key]
        values_data.append(val)
    cols = [i for i in range(1,len(values_data[0]))]
    cols.append('aid')
    data_df = pd.DataFrame(data=values_data,columns=cols)

    return data_df

def combine_to_list(row):
    ans = []
    keys = row.index.tolist()
    keys.remove('aid')
    for key in keys:
        ans.append(int(row[key]))

    return ans

if __name__ == '__main__':
    dataset = 'phone'
    data_path = 'datasets'
    embedding_file = os.path.join(data_path,dataset,'embeddings_single.pth') # 预训练获得的item embedding
    corpus_file = os.path.join(data_path,dataset,'corpus_512.json')
    with open(corpus_file,'r') as f:
        corpus = json.load(f)

    accelerator = accelerate.Accelerator()
    embedding_dict = torch.load(embedding_file,map_location=accelerator.device)

    # content_embed = embedding_dict['content']
    colla_embed = embedding_dict['colla']
    solution = Solution(colla_embed,length=3,cluster=32)
    # start
    all_item_ids = np.arange(1, solution.total_num + 1) # item:1000 [1,2,3，……1000]
    solution.kmeans_cluster(1,all_item_ids)


    gid_df = get_gid_df(solution.all_dict)

    num_cols = gid_df.columns.tolist()
    num_cols.remove('aid')
    gid_df[solution.length] = gid_df.groupby(num_cols).cumcount()+1

    gid_df['cid'] = gid_df.apply(combine_to_list,axis=1)
    new_cids = gid_df.set_index('aid')['cid'].to_dict()
    for key,value in corpus.items():
        value.update({
            'cid': new_cids[int(key)]
        })

    outpath = os.path.join(data_path,dataset)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = os.path.join(outpath,'corpus_512_test.json') # batch1
    with open(outfile,'w') as f:
        json.dump(corpus,f)



