import torch
import numpy as np

strict_list = None

def get_pos_index(predict,target):
    pos_index = []
    for u, u_list in enumerate(predict):
        u_result = []
        target[u] = [item for item in target[u] if item != 0]
        for i in u_list:
            if i != 0 and i in target[u]:
                u_result.append(1)
            else:
                u_result.append(0)

        u_result.append(len(target[u]))
        pos_index.append(u_result)
    return pos_index

def compute_all_matrics(matrics,topk,topk_idx,pos_len_list,
                        prediction_list=None,
                        **kwargs):
    result_dict = {}
    for matric in matrics:
        if matric.lower() == 'ndcg':
            ans_dict = compute_ndcg(topk_idx,pos_len_list,topk)
            result_dict.update(ans_dict)
        elif matric.lower() == 'recall':
            ans_dict = compute_recall(topk_idx,pos_len_list,topk)
            result_dict.update(ans_dict)

        elif matric.lower() == 'itemcoverage':
            ans_dict = comput_itemcoverage(prediction_list,topk=topk,**kwargs)
            result_dict.update(ans_dict)
        else:
            print('The matric of {} has not been implemented yet.')

    return result_dict

def comput_itemcoverage(predict_list,tot_item_num,topk,**kwargs):
    metric_dict = {}
    for k in topk:
        key = "{}@{}".format('itemcoverage',k)
        now_list = predict_list[:,:k]
        unique_count = np.unique(now_list).shape[0]
        value = unique_count / tot_item_num
        metric_dict[key] = value

    return metric_dict

def compute_ndcg(topk_idx, pos_len_list,topk):
    pos_index = topk_idx.to(torch.bool).numpy()
    pos_len = pos_len_list.squeeze(-1).numpy()

    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float64)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float64)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    avg_result = result.mean(axis=0) # 按照位置取结果就可以
    ans = {}
    for k in topk:
        key = "NDCG@{}".format(k)
        ans[key] = avg_result[k-1]

    return ans

def compute_recall(topk_idx, pos_len_list,topk):
    pos_index = topk_idx.to(torch.bool).numpy()
    pos_len = pos_len_list.squeeze(-1).numpy()

    result = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    avg_result = result.mean(axis=0)
    ans = {}
    for k in topk:
        key = "recall@{}".format(k)
        ans[key] = avg_result[k-1]

    return ans


def output_preprocess(cid_list: list):
    new_cid_list = []
    for u in cid_list:
        for idx, k in enumerate(u):
            u[idx] = [str(i) for i in k]
        u_list = ['_'.join(k) for k in u]
        new_cid_list.append(u_list)

    return new_cid_list

def get_cid_dict(item_corpus):
    cid2atom = {}
    for key, value in item_corpus.items():
        cid = [str(i) for i in value['cid']]
        cid = '_'.join(cid)
        cid2atom[cid] = int(key)

    return cid2atom


def decode_cid2atomid(cid_list: list, cid_dict, unknown_id):

    result_list = []
    for user in cid_list:
        predict = [cid_dict.get(item, unknown_id) for item in user]
        result_list.append(predict)

    return result_list


def cut_intered_prediction(prediction,inter_list,max_topk,unknown_id):
    new_prediction = []
    user_unknown = 0
    item_unknown = 0
    item_ungenerate = 0
    for u,pred in enumerate(prediction):
        inter = [i for i in inter_list[u] if i !=0]
        ans = [i for i in pred if i not in inter]

        if len(ans) < max_topk:
            ori_len = len(ans)
            ans += [unknown_id] * (max_topk - len(ans))
            item_ungenerate += (max_topk - ori_len)
        ans = ans[:max_topk]
        new_prediction.append(ans)
        if unknown_id in ans:
            user_unknown += 1
            item_unknown += ans.count(unknown_id)

    return new_prediction,user_unknown,item_unknown,item_ungenerate

