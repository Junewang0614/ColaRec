import sys
sys.path.append('../')
import os
import torch
from torch.optim import AdamW
from accelerate import Accelerator
import math
from datetime import datetime

from other.new_utils import (set_seed,get_logger,save_load_ckpt,compute_bpr_loss,
                             save_checkpoint,get_lastpos_state,get_generate_acc)

from src.tokenization import V4T5Tokenizer # tokenizer

from src.models import V4T5smallModel,CIDT5RecConfig # model
from other.config_class import DataFileConfig,DataProcessConfig
from other.configs import v4T5small_train

from src.datasets import get_train_dataloader_augitem
from src.datasets import get_valid_dataloader_timeorder

import argparse
parser = argparse.ArgumentParser(description='Process some hype-parameters.')
parser.add_argument('--info', default='v4_long_sport_userid1',help='other information')
parser.add_argument('--eval_mode', default='latest',help='the mode for eval data process')
parser.add_argument('--sample_user_num', type=int,default=1,help='the number of sampling user')
parser.add_argument('--item_position',action='store_true',help='whether use position embedding or not')
parser.add_argument('--dataset',default='beauty',help='dataset name')
parser.add_argument('--cid_token_num', type=int,default=32,help='cid token number')
parser.add_argument('--generate_lamda', type=float,default=0.08,help='the weight of generative loss')
parser.add_argument('--content_cl_lamda', type=float,default=0.1,help='the weight of cl bpr loss')

parser.add_argument('--batch_size',type=int,default=128,help='training batch size')
parser.add_argument('--seq_type',default='short',help='seq length type')
parser.add_argument('--n_epochs',type=int,default=60,help='training epoch')

args = parser.parse_args()

def main():
    accelerator = Accelerator()
    train_config = v4T5small_train

    train_config.update(args.__dict__)
    data_config = DataFileConfig(args.dataset)
    set_seed(train_config['seed'])
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    data_config.save_path = os.path.join(data_config.save_path,train_config['info']+'_{}'.format(time_str))
    logger = get_logger(data_config.save_path)
    logger.info(train_config)

    pro_config = DataProcessConfig(mode=args.eval_mode,sample_user_num=args.sample_user_num,
                                   seq_type=train_config['seq_type'])
    pro_config.updata_for_type()

    tokenizer = V4T5Tokenizer.from_pretrained(data_config.pretrain, config=pro_config)
    train_dataloader = get_train_dataloader_augitem(data_config,pro_config,tokenizer,
                                            train_config['batch_size'],num_workers=train_config['num_workers'])
    _,valid_dataset = get_valid_dataloader_timeorder(data_config,pro_config,tokenizer,batch_size=train_config['eval_batch_size'])

    pro_config.total_item_num = len(valid_dataset.item_corpus.keys())
    pro_config.total_user_num = valid_dataset.total_user_num # user大小
    pro_config.unknown_id = pro_config.total_item_num + 1
    con_dict = dict(
        item_num=pro_config.total_item_num,
        user_num = pro_config.total_user_num,
        cid_token_num=args.cid_token_num,
        code_num=3,
        code_length = 3,
        max_info_len = pro_config.max_infor_len,
        item_position = train_config['item_position']
    )
    config = CIDT5RecConfig.from_pretrained(data_config.pretrain,new_dict=con_dict)
    logger.info(config)
    logger.info(pro_config.__dict__)

    # model
    model = V4T5smallModel(config)
    model = save_load_ckpt(model, data_config.pretrain)

    optimizer = AdamW(model.parameters(), lr=train_config['total_lr'])
    logger.info(model)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)
    print('the device is:', train_dataloader.device)

    cnter = 0

    for epoch in range(train_config['n_epochs']):
        model.train()
        total_index_loss = 0
        total_retrieval_loss = 0
        total_bpr_loss = 0
        total_loss = 0
        total_acc = 0
        total_index_acc = 0
        total_content_cl = 0

        for batch in train_dataloader:
            user_input_dict, pos_input_dict, neg_input_dict, labels, neg_labels, aug_input_dict = batch

            # index部分注意，要说明是item
            pos_index_output = model.forward_train(**pos_input_dict,**labels,ty='item')
            neg_index_output = model.forward_train(**neg_input_dict,**neg_labels,ty='item')
            index_loss = pos_index_output.loss + neg_index_output.loss

            # retrieval部分
            retrieval_output = model.forward_train(**user_input_dict,**labels,ty='user')
            retrieval_loss = retrieval_output.loss
            generative_loss = index_loss + retrieval_loss

            neg_hidden_state = get_lastpos_state(neg_index_output.encoder_last_hidden_state,neg_input_dict['attention_mask'])
            pos_hidden_state = get_lastpos_state(pos_index_output.encoder_last_hidden_state,pos_input_dict['attention_mask'])
            user_hidden_state = get_lastpos_state(retrieval_output.encoder_last_hidden_state,user_input_dict['attention_mask'])

            bpr_loss = compute_bpr_loss(model.bpr_loss, user_hidden_state, pos_hidden_state, neg_hidden_state)

            # 1.获得aug item的状态
            aug_index_output = model.get_encoder_state(**aug_input_dict,**labels,ty='item')
            aug_hidden_state = get_lastpos_state(aug_index_output.encoder_last_hidden_state,aug_input_dict['attention_mask'])
            # 2. 计算loss 这里基底是pos state,正例是aug_state,负例是neg_state
            cl_loss = compute_bpr_loss(model.bpr_loss,pos_hidden_state,aug_hidden_state,neg_hidden_state)

            if 'generate_lamda' in train_config.keys():
                all_loss = train_config['generate_lamda'] * generative_loss + bpr_loss \
                           + train_config['content_cl_lamda'] * cl_loss
            else:
                all_loss = generative_loss + bpr_loss \
                           + train_config['content_cl_lamda'] * cl_loss

            accelerator.backward(all_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            optimizer.zero_grad()

            total_index_loss += index_loss.item()
            total_retrieval_loss += retrieval_loss.item()
            total_bpr_loss += bpr_loss.item()
            total_loss += all_loss.item()
            total_content_cl += cl_loss.item()

            # retrieval的acc
            retrieval_acc = get_generate_acc(retrieval_output.logits,labels['labels'])
            total_acc += retrieval_acc
            # index的acc
            pos_index_acc = get_generate_acc(pos_index_output.logits,labels['labels'])
            neg_index_acc = get_generate_acc(neg_index_output.logits,neg_labels['labels'])
            total_index_acc += (pos_index_acc + neg_index_acc)

            if cnter % train_config['print_steps'] == 0:
                print('the current batch {}, loss:{},retrieval acc:{},index acc:{}'.format(cnter, all_loss.item(),
                                                                              retrieval_acc,(pos_index_acc + neg_index_acc)))
                logger.info('the current batch {}, loss:{},retrieval acc:{},index acc:{}'.format(cnter, all_loss.item(),
                                                                              retrieval_acc,(pos_index_acc + neg_index_acc)))
            cnter += 1


        print('EPOCH {}, total index loss:{:.4f}, total retrieval loss:{:.4f}, total bpr loss:{:.4f}'.format(epoch,
                                                                                                             total_index_loss, total_retrieval_loss,
                                                                                                             total_bpr_loss))
        print('EPOCH {}, total cl loss:{:.4f}, total loss:{:.4f}'.format(epoch,total_content_cl,total_loss))
        print('EPOCH {}, index acc:{:.4f}, retrieval acc:{:.4f}'.format(epoch, total_index_acc / len(train_dataloader),
                                                                            total_acc / len(train_dataloader)))

        logger.info('EPOCH {}, total index loss:{:.4f}, total retrieval loss:{:.4f}, total bpr loss:{:.4f}'.format(epoch,
                                                                                                             total_index_loss, total_retrieval_loss,
                                                                                                             total_bpr_loss))
        logger.info('EPOCH {}, total cl loss:{:.4f}, total loss:{:.4f}'.format(epoch,total_content_cl,total_loss))
        logger.info('EPOCH {}, index acc:{:.4f}, retrieval acc:{:.4f}'.format(epoch, total_index_acc / len(train_dataloader),
                                                                            total_acc / len(train_dataloader)))

        save_path = os.path.join(data_config.save_path, 'epoch{}'.format(epoch + 1))
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_checkpoint(save_path, unwrapped_model, accelerator)

    logger.info(30 * '+')
    logger.info('the model saving path is {}'.format(data_config.save_path))


if __name__ == '__main__':
    main()

