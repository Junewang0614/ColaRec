# 要基于T5
import torch
from transformers import T5Tokenizer

class T5TokenizerV1(T5Tokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,config=None,new_vob=None):
        cls.config = config # 相较于原本的tokenizer，会把config加进来

        # 如果有新的单词，会添加到tokenizer中
        tokenizer = super().from_pretrained(pretrained_model_name_or_path)
        if new_vob is not None:
            tokenizer.add_tokens(sorted(new_vob))
        return tokenizer

    def __call__(self,items,pad_to_max=False,return_tensor=False,is_label=False):

        if len(items) > 0 and isinstance(items[0],list):
            if is_label:
                inputs = self.batch_encode_semid(items)
            else:
                inputs = self.batch_encode(items,pad_to_max=pad_to_max)

        else:
            if is_label:
                inputs = self.encode_semid(items)
            else:
                inputs = self.encode(items)

        if return_tensor:
            for k,v in inputs.items():
                inputs[k] = torch.LongTensor(v)

        return inputs


    def encode(self,items):
        items = items[:self.config.max_item_num]

        input_ids = []
        for idx,item_info in enumerate(items):
            info_ids = self.convert_tokens_to_ids(self.tokenize(item_info))
            info_ids = info_ids[:self.config.max_infor_len]

            input_ids += info_ids

        # 切
        input_ids = input_ids[:self.config.max_token_num-1] # 加eos
        input_ids.append(self.eos_token_id)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask
        }


    def batch_encode(self,item_batch,pad_to_max=False):
        item_batch = [self.encode(items) for items in item_batch]

        return self.padding(item_batch,pad_to_max)


    def padding(self,item_batch,pad_to_max):
        if pad_to_max:
            max_length = self.config.max_token_num
        else:
            max_length = max([len(items["input_ids"]) for items in item_batch])

        batch_input_ids = []
        batch_attention_mask = []

        for items in item_batch:
            input_ids = items["input_ids"]
            attention_mask = items["attention_mask"]

            length_to_pad = max_length - len(input_ids)

            input_ids += [self.pad_token_id] * length_to_pad # input_ids
            attention_mask += [0] * length_to_pad # attention mask

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        }

    def batch_encode_semid(self, batch_item):
        item_batch = [self.encode_semid(item) for item in batch_item]

        return self.semid_padding(item_batch)

    def encode_semid(self, item):
        item = item[0]
        label_ids = self.convert_tokens_to_ids(self.tokenize(item))
        label_ids.append(self.eos_token_id)

        return {'labels': label_ids}

    def semid_padding(self, batch_item):
        max_length = max([len(item["labels"]) for item in batch_item])
        batch_labels = []

        for item in batch_item:
            labels = item['labels']
            length_to_pad = max_length - len(labels)

            labels += [-100] * length_to_pad
            batch_labels.append(labels)

        return {
            'labels': batch_labels
        }

class V4T5Tokenizer(T5TokenizerV1):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,config=None,new_vob=None):
        # cls.config = config # 相较于原本的tokenizer，会把config加进来
        # 如果有新的单词，会添加到tokenizer中
        tokenizer = super().from_pretrained(pretrained_model_name_or_path,config=config)
        tokenizer.atom_pad = config.atom_pad
        tokenizer.atom_pad_id = tokenizer.convert_tokens_to_ids(config.atom_pad)
        tokenizer.atom_user_pad = config.atom_user_pad
        tokenizer.atom_user_pad_id = tokenizer.convert_tokens_to_ids(config.atom_user_pad)
        if new_vob is not None:
            tokenizer.add_tokens(sorted(new_vob))
        return tokenizer

    def __call__(self,items,pad_to_max=False,return_tensor=False,ty='user'):
        assert ty in ['user','item']

        if len(items) > 0 and isinstance(items[0],list):
            inputs = self.batch_encode(items,pad_to_max=pad_to_max,ty=ty)

        else:
            inputs = self.encode(items,ty=ty)

        if return_tensor:
            for k,v in inputs.items():
                inputs[k] = torch.LongTensor(v)

        return inputs


    def encode(self,items,ty='user'):
        items = items[:self.config.max_item_num]
        input_ids = [self.pad_token_id]
        input_positions = [1]
        input_index = []
        for idx,item_info in enumerate(items):

            info_ids = self.convert_tokens_to_ids(self.tokenize(item_info))
            info_ids = [self.atom_pad_id] + info_ids
            info_ids = info_ids[:self.config.max_infor_len]

            input_index.append(len(input_ids))
            input_position = list(range(3,len(info_ids)+3))
            input_ids += info_ids
            input_positions += input_position

        input_ids = input_ids[:self.config.max_token_num-1]
        input_positions = input_positions[:self.config.max_token_num-1]

        if ty == 'item':
            start = len(input_ids)
            user_index = list(range(start,start + self.config.sample_user_num))
            input_ids += [self.atom_user_pad_id]*self.config.sample_user_num
            input_positions += [2] * self.config.sample_user_num

        input_ids.append(self.eos_token_id)
        input_positions.append(0)
        item_num = input_ids.count(self.atom_pad_id)
        input_index = input_index[:item_num]
        attention_mask = [1] * len(input_ids)

        if ty == 'item':
            return {
                "content_input_ids": input_ids,
                "atom_index": input_index,
                "attention_mask": attention_mask,
                "input_positions":input_positions,
                "user_atom_index":user_index
            }
        else:
            return {
                "content_input_ids": input_ids,
                "atom_index": input_index,
                "attention_mask": attention_mask,
                "input_positions": input_positions,
            }

    def padding(self,item_batch,pad_to_max):
        if pad_to_max:
            max_length = self.config.max_token_num
        else:
            max_length = max([len(items["content_input_ids"]) for items in item_batch])

        max_index_length = max([len(items["atom_index"]) for items in item_batch])

        batch_input_ids = []
        batch_input_index = []
        batch_attention_mask = []
        batch_input_positions = []

        for items in item_batch:
            input_ids = items["content_input_ids"]
            input_index = items['atom_index']
            attention_mask = items["attention_mask"]
            input_positions = items["input_positions"]

            length_to_pad = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * length_to_pad # input_ids
            attention_mask += [0] * length_to_pad # attention mask
            input_positions += [0] * length_to_pad

            index_length_to_pad = max_index_length - len(input_index)
            input_index += [0] * index_length_to_pad

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_input_index.append(input_index)
            batch_input_positions.append(input_positions)

        return {
            "content_input_ids":batch_input_ids,
            "atom_index":batch_input_index,
            "attention_mask":batch_attention_mask,
            "input_positions": batch_input_positions
        }

    def batch_encode(self,item_batch,pad_to_max=False,ty='user'):
        item_batch = [self.encode(items,ty=ty) for items in item_batch]

        ans_dict = self.padding(item_batch,pad_to_max)
        if ty == 'item':
            batch_user_index = [items['user_atom_index'] for items in item_batch]
            ans_dict.update({
                'user_atom_index':batch_user_index
            })

        return ans_dict