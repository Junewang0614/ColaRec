import json
import re
import pandas as pd


def trans_cid2int_list(item_corpus):
    pattern = re.compile(r'\d{1,3}')
    for key,value in item_corpus.items():
        old_cid = value['cid']
        new_cid = [int(i) for i in pattern.findall(old_cid)]
        item_corpus[key]['cid'] = new_cid
    return item_corpus


def get_inter_list(file,with_neg=False) -> list:
    assert '.csv' in file, 'the inter file type must be csv.'
    inter_data = pd.read_csv(file)
    if with_neg:
        assert len(inter_data.columns) >= 3
        inter_list = list(zip(inter_data['user_id'].tolist(), inter_data['item_id'].tolist(),inter_data['neg_id'].tolist()))
    else:
        inter_list = list(zip(inter_data['user_id'].tolist(), inter_data['item_id'].tolist()))

    return inter_list


def get_user_interlist(file,return_list = False,time_order=True):
    assert '.csv' in file,'the inter file type must be csv.'
    user_inter_dict = {}
    inter_data = pd.read_csv(file)
    if time_order:
        assert 'time' in inter_data.columns,'time_order mode need the data file have time col.'
    user_list = inter_data['user_id'].unique().tolist()
    user_inter_list = []
    for user in user_list:
        if time_order:
            inter_list = inter_data[inter_data['user_id'] == user].sort_values(['time'])['item_id'].tolist()
        else:
            inter_list = inter_data[inter_data['user_id'] == user]['item_id'].tolist()
        user_inter_dict[user] = inter_list
        user_inter_list.append((user,inter_list))

    if return_list:
        return user_inter_dict,user_inter_list
    else:
        return user_inter_dict


def get_item_interlist(file,return_list=False):
    assert '.csv' in file, 'the inter file type must be csv.'
    item_inter_dict = {}
    inter_data = pd.read_csv(file)
    item_list = inter_data['item_id'].unique().tolist()
    item_inter_list = []

    for item in item_list:
        inter_list = inter_data[inter_data['item_id'] == item]['user_id'].tolist()

        item_inter_dict[item] = inter_list
        item_inter_list.append((item,inter_list))

    if return_list:
        return item_inter_dict,item_inter_list
    else:
        return item_inter_dict


def get_item_content(file,
                     keys=['title','brand','categories'],
                     keep_keys=True) -> dict:
    assert '.json' in file,'The raw content file must be a json file'
    content_dict = {}
    with open(file,'r') as f:
        raw_content_data = json.load(f)

    for item in raw_content_data:
        atom_id = int(item['item_id'])
        item_info = ''
        for key in keys:
            info = item.get(key, '')
            if info == '':
                continue
            if isinstance(info, list):  # categories的问题
                info = ' '.join(info[0][1:])
            if keep_keys:
                item_info = ' '.join([item_info, key, info])
            else:
                item_info = ' '.join([item_info, info])
            item_info = item_info.strip()

        content_dict[atom_id] = item_info

    return content_dict