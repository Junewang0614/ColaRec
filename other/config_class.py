import os
from dataclasses import dataclass

# data_config
class DataFileConfig:
    def __init__(self,dataset,pca=False,no_content=False,new_words=False,
                 diff_gid=None):
        self.dataset = dataset
        # if path is None:
        self.path = "./" # 主目录
        self.save_path = "save_models"
        self.dataset_path = os.path.join(self.path, 'datasets',self.dataset)
        self.pretrain = "pretrained/t5-small/" # pretrain model path
        self.save_path = os.path.join(self.save_path, 'models')
        if new_words:
            self.new_token = os.path.join(self.dataset_path,'new_tokens.json')
        if diff_gid is not None:
            assert diff_gid in ['random','bert']
            if diff_gid == 'random':
                self.corpus_file = os.path.join(self.dataset_path, 'corpus_512_random.json')
            elif diff_gid == 'bert':
                self.corpus_file = os.path.join(self.dataset_path, 'corpus_512_semi.json')

        elif no_content:
            self.corpus_file = os.path.join(self.dataset_path,'corpus_512_no_content.json')
        else:
            self.corpus_file = os.path.join(self.dataset_path,'corpus_512.json')

# pro_config
@dataclass
class DataProcessConfig:
    atom_pad:str = '<extra_id_0>'
    atom_user_pad:str = '<extra_id_1>'
    cid_item_pad:str = '<extra_id_2>'

    total_item_num:int=0# 从valid dataset更新
    total_user_num: int = 0  # 从valid dataset更新
    unknown_id:int=0
    sample_user_num:int=1
    mode:str = 'latest'
    seq_type:str = 'long'

    max_item_num: int = 40  # 20
    max_token_num: int = 500  # 256
    max_infor_len: int = 50
    sample_item_num: int = 30  # 10

    id_len: int = 3

    neg_p: float = 0.2
    neg_ty:int = 2

    consim_k:int = 500
    consim_file:str = 'similar_items_lam087.json'

    def updata_for_type(self):
        if self.seq_type == 'mlshort':
            self.max_item_num: int = 40
            self.max_token_num: int = 256
            self.max_infor_len: int = 30   # 30
            self.sample_item_num: int = 20

        elif self.seq_type == 'micshort':
            self.max_item_num: int = 20
            self.max_token_num: int = 256
            self.max_infor_len: int = 35  #  30
            self.sample_item_num: int = 15
        elif self.seq_type != 'long':
            self.max_item_num: int = 20
            self.max_token_num: int = 256
            self.max_infor_len: int = 30
            self.sample_item_num: int = 10



if __name__ == '__main__':
    pro_config = DataProcessConfig(seq_type='long')
    print(pro_config)
    pro_config = DataProcessConfig(seq_type='short')
    pro_config.updata_for_type()
    print(pro_config)