import sys
sys.path.append('../')
import os
import torch
from accelerate import Accelerator
import math
from datetime import datetime

from other.new_utils import set_seed,get_logger

from src.tokenization import V4T5Tokenizer

from src.models import V4T5smallModel,CIDT5RecConfig
from other.config_class import DataFileConfig,DataProcessConfig
from other.configs import v4T5small_train
from other.new_utils import PrefixTree

from src.datasets import get_valid_dataloader_timeorder # valid dataloader
from src.evaluation import constrained_eval_process # evaluation

import argparse
parser = argparse.ArgumentParser(description='Process some hype-parameters.')
# epoch
parser.add_argument('--n_epochs', type=int,default=100,help='the number of training epochs')
# parser.add_argument('--batch_size', type=int,default=128,help='the number of training batch')
parser.add_argument('--eval_batch_size', type=int,default=4,help='the number of training batch')
parser.add_argument('--info', default='v4_no_position_user_atomids',help='other information')
parser.add_argument('--eval_mode', default='latest',help='the mode for eval data process')
parser.add_argument('--eval_model_mkdir',
                    default="",
                    help='the models path to check')
parser.add_argument('--dataset',default='beauty',help='dataset name')
parser.add_argument('--seq_type',default='short',help='seq length type')
parser.add_argument('--start',type=int,default=15,help='the number of epoch to start eval')
parser.add_argument('--row',type=float,default=1,help='generation row')
parser.add_argument('--no_content',action='store_true',help='whether not use content')
args = parser.parse_args()

def main():
    accelerator = Accelerator()
    train_config = v4T5small_train
    train_config.update(args.__dict__)
    data_config = DataFileConfig(args.dataset,no_content=train_config['no_content'])

    set_seed(train_config['seed'])
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    data_config.save_path = os.path.join(args.eval_model_mkdir,'best_model_{}-{}_{}'.format(args.start+1,
                                                                                            args.start + 1+args.n_epochs,
                                                                                            time_str))
    logger = get_logger(data_config.save_path)
    logger.info(train_config)

    pro_config = DataProcessConfig(mode=args.eval_mode,seq_type=train_config['seq_type'])
    pro_config.updata_for_type()
    tokenizer = V4T5Tokenizer.from_pretrained(data_config.pretrain, config=pro_config)
    valid_dataloader, valid_dataset = get_valid_dataloader_timeorder(data_config,pro_config,tokenizer,batch_size=train_config['eval_batch_size'])
    test_dataloader,_ = get_valid_dataloader_timeorder(data_config,pro_config,tokenizer,
                                                       batch_size=train_config['eval_batch_size'],
                                                       split='test')

    pro_config.total_item_num = len(valid_dataset.item_corpus.keys())
    pro_config.total_user_num = len(valid_dataset)
    pro_config.unknown_id = pro_config.total_item_num + 1
    con_dict = dict(
        item_num=pro_config.total_item_num,
        user_num = pro_config.total_user_num,
        cid_token_num=32,
        code_num=3,
        code_length = 3,
        max_info_len = pro_config.max_infor_len,
        item_position = train_config['item_position']
    )

    cid_list = [[tokenizer.pad_token_id] + value['cid'] for value in
                valid_dataset.item_corpus.values()]
    prefixtree = PrefixTree()
    prefixtree.set_all(cid_list)

    config = CIDT5RecConfig.from_pretrained(
            os.path.join(args.eval_model_mkdir,'epoch1'),
             # new_dict=con_dict
        )
    logger.info(config)
    logger.info(pro_config.__dict__)

    model = V4T5smallModel(config)

    best_result = None
    generate_config = {
        'num_beams':train_config['num_beams'],
        'num_return_sequences':train_config['num_return_sequences'],
        'max_new_tokens':config.code_length,
        'eos_token_id':config.code_size + 1,
        'row':train_config['row']
    }
    valid_dataloader,test_dataloader = accelerator.prepare(
        valid_dataloader,test_dataloader)
    print('the device is:', valid_dataloader.device)
    logger.info(generate_config)

    stop = 0
    ori_eval_step = train_config['eval_steps']


    start = args.start
    for epoch in range(start,start+train_config['n_epochs']+1):
        subpath = 'epoch'+str(epoch+1)
        model_path = os.path.join(args.eval_model_mkdir,subpath)
        if not os.path.exists(model_path):
            logger.info('There is no subpath {}'.format(model_path))
            continue

        logger.info('Now is evaling the model in path {}'.format(model_path))
        # 加载模型
        model = model.from_pretrained(model_path)
        torch.cuda.empty_cache()
        model = model.to(accelerator.device)
        # 模型验证
        model.eval()
        with torch.no_grad():
            eval_results= constrained_eval_process(model,valid_dataloader,
                                              valid_dataset.item_corpus,
                                              config=train_config, pro_config=pro_config,
                                              generate_config=generate_config,
                                                prefix_allowed_tokens_fn=prefixtree)

        if best_result is None or eval_results[train_config['key_metric']] > best_result[train_config['key_metric']]:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(data_config.save_path, save_function=accelerator.save)

            # 更新指标
            best_result = eval_results.copy()
            # 记录指标
            logger.info(
                'SUBPATH {}, ======================== The best results ========================'.format(
                    subpath))
            logger.info(best_result)
            logger.info('============================================================')
            stop = 0
        else:
            stop += 1

        if stop > train_config['early_stop']:
            logger.info('=================early stop at EPOCH {}======================'.format(epoch+1))
            logger.info(best_result)
            break

    logger.info(30*'+')
    logger.info('Evaling is over,the test result:')

    model = model.from_pretrained(data_config.save_path)
    model = accelerator.prepare(model)
    model.eval()
    with torch.no_grad():
        test_results = constrained_eval_process(model, test_dataloader, valid_dataset.item_corpus,
                                          config=train_config,pro_config=pro_config,
                                          generate_config=generate_config,
                                                prefix_allowed_tokens_fn=prefixtree)
        logger.info('==============now the row is:{}================'.format(generate_config['row']))
        logger.info(test_results)
        generate_config['row'] = 0.9
        test_results = constrained_eval_process(model,test_dataloader,valid_dataset.item_corpus,
                                          config=train_config, pro_config=pro_config,
                                          generate_config=generate_config,prefix_allowed_tokens_fn=prefixtree)
        logger.info('==============now the row is:{}================'.format(generate_config['row']))
        logger.info(test_results)


if __name__ == '__main__':
    main()
