import torch
from tqdm import tqdm
from other.evaluation import *
import numpy as np

@torch.no_grad()
def constrained_eval_process(model,eval_dataloader,
                           item_corpus,
                           config,pro_config,generate_config,
                            prefix_allowed_tokens_fn = None,
                           show_process = False):
    cid2atom = get_cid_dict(item_corpus)
    min_beam = max(config['topk']) + pro_config.max_item_num
    if generate_config.get('num_return_sequences',0) < min_beam:
        generate_config['num_beams'] = min_beam
        generate_config['num_return_sequences'] = min_beam

    # prefix
    generate_config['prefix_allowed_tokens_fn'] = prefix_allowed_tokens_fn
    print(generate_config['prefix_allowed_tokens_fn'])
    predict_matrix = None
    total_user_unknown = 0
    total_item_unknown = 0
    total_item_ungener = 0
    predict_list = []

    if show_process:
        print(generate_config)
    eval_dataloader = tqdm(eval_dataloader) if show_process else eval_dataloader

    for batch in eval_dataloader:
        user_input_dict, user_target, user_intered = batch
        batch_size = len(user_target)
        outputs = model.predict(generate_config, **user_input_dict)
        outputs = outputs.reshape([batch_size, -1, outputs.shape[-1]])
        outputs = outputs[:, :, 1:].tolist()
        new_cid_list = output_preprocess(outputs)
        prediction = decode_cid2atomid(new_cid_list, cid2atom, pro_config.unknown_id)
        prediction, user_unknown, item_unknown,item_ungenerate = cut_intered_prediction(prediction, user_intered,
                                                                        max(config['topk']), pro_config.unknown_id)
        total_user_unknown += user_unknown
        total_item_unknown += item_unknown
        total_item_ungener += item_ungenerate
        predict_list += prediction

        # 构造矩阵
        pos_index = get_pos_index(prediction,user_target)

        if predict_matrix is None:
            predict_matrix = torch.tensor(pos_index,dtype=torch.int32)
        else:
            pos_index = torch.tensor(pos_index,dtype=torch.int32)
            predict_matrix = torch.cat((predict_matrix,pos_index),dim=0)

    predict_list = np.array(predict_list)
    topk_idx, pos_len_list = torch.split(predict_matrix,
                                         [max(config['topk']), 1], dim=1)

    result_dict = compute_all_matrics(config['metrics'], config['topk'],
                                      topk_idx, pos_len_list,
                                      prediction_list=predict_list, tot_item_num=len(item_corpus.keys()))

    result_dict.update({
        'generate_unknown_user_num':total_user_unknown,
        'generate_total_unknown_item':total_item_unknown,
        'generate_not_enough_item_num':total_item_ungener,
    })
    return result_dict

