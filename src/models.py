import torch
import warnings
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import copy
from typing import Optional, Tuple, Union
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Stack,
    T5Block,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5LayerFF,
    T5LayerCrossAttention,
    T5PreTrainedModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput
)
from src.other_modules import BPRLoss

class CIDT5RecConfig(T5Config):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,new_dict=None):
        t5_ori = super().from_pretrained(pretrained_model_name_or_path)
        if new_dict is not None:
            t5_ori.__dict__.update(new_dict)
            t5_ori.code_size = t5_ori.cid_token_num

        return t5_ori


    def __init__(self,item_num=1000,
                 cid_token_num = 32,
                 code_num = 1,
                 code_length=3,
                 **kwargs):

        super().__init__(**kwargs)

        self.item_num = item_num
        self.cid_token_num = cid_token_num
        self.code_num = code_num 
        self.code_size = self.cid_token_num
        self.code_length = code_length

    def update(self,newdict):
        self.__dict__.update(newdict)
        self.code_size = self.cid_token_num

class V4T5smallModel(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]

    def __init__(self,config):

        super(V4T5smallModel, self).__init__(config)
        assert config.code_num == config.code_length
        # note:=========非t5本身的层==========
        self.atomid_embed = nn.Embedding(config.item_num + 1, config.d_model)
        self.codebook_num = config.code_num
        self.cid_embed_list = nn.ModuleList(
            [
                nn.Embedding(config.cid_token_num+1,config.d_model)
                for i in range(self.codebook_num)
            ]
        )
        self.centroids = nn.ModuleList([
            nn.Linear(config.d_model,config.cid_token_num+1)
            for _ in range(self.codebook_num)
        ])
        # note:=============================

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.prompt = nn.Embedding(2,config.d_model)
        self.atomid_user_embed = nn.Embedding(config.user_num+1,config.d_model)
        if config.item_position:
            self.info_position = nn.Embedding(config.max_info_len+10,config.d_model)
        else:
            self.info_position = None

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        self.bpr_loss = BPRLoss()

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def get_encoder_state(self,
                      content_input_ids: Optional[torch.LongTensor] = None,
                      atom_input_ids: Optional[torch.LongTensor] = None,
                      atom_index: Optional[torch.LongTensor] = None,
                      attention_mask: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      user_atom_ids: Optional[torch.LongTensor] = None, 
                      user_atom_index: Optional[torch.LongTensor] = None, 
                      input_positions: Optional[torch.LongTensor] = None,
                      output_attentions: Optional[bool] = None,
                      output_hidden_states: Optional[bool] = None,
                      use_cache: Optional[bool] = None,
                      return_dict: Optional[bool] = None,
                      ty='user'):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # task token
        batch_size = atom_input_ids.shape[0]
        device = atom_input_ids.device
        prompt_embed = self.get_prompt_embed(batch_size,device,ty)

        assert atom_index.shape == atom_input_ids.shape
        contend_embed = self.shared(content_input_ids)
        atomid_embed = self.atomid_embed(atom_input_ids)
        embed_shape = atomid_embed.shape[-1]
        exindex = atom_index.unsqueeze(-1).expand(*atom_index.shape,embed_shape)
        encoder_input_embed = contend_embed.scatter(dim=1,index=exindex,src=atomid_embed)
        encoder_input_embed[:, 0] = prompt_embed

        if ty == 'item' and user_atom_ids is not None:
            assert user_atom_ids.shape == user_atom_index.shape
            user_atomid_embed = self.atomid_user_embed(user_atom_ids)
            user_exindex = user_atom_index.unsqueeze(-1).expand(*user_atom_index.shape,embed_shape)
            encoder_input_embed = encoder_input_embed.scatter(dim=1,index=user_exindex,src=user_atomid_embed)

        if self.info_position is not None and input_positions is not None:
            assert input_positions.shape == encoder_input_embed.shape[:2]
            input_position_embed = self.info_position(input_positions)
            encoder_input_embed += input_position_embed

        encoder_outputs = self.encoder(
            inputs_embeds=encoder_input_embed,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return Seq2SeqLMOutput(
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def forward_train_useraid(self,
                            content_input_ids: Optional[torch.LongTensor] = None,
                              attention_mask: Optional[torch.FloatTensor] = None,
                              labels: Optional[torch.LongTensor] = None,
                              output_attentions: Optional[bool] = None,
                              output_hidden_states: Optional[bool] = None,
                              use_cache: Optional[bool] = None,
                              return_dict: Optional[bool] = None,
                              ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        batch_size = content_input_ids.shape[0]
        device = content_input_ids.device
        prompt_embed = self.get_prompt_embed(batch_size,device,'user')
        encoder_input_embed = self.atomid_user_embed(content_input_ids)
        assert prompt_embed.shape == encoder_input_embed.shape
        encoder_input_embed = torch.cat([prompt_embed,encoder_input_embed],dim=1)

        # 2. encoder过
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_input_embed,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 3. decoder过
        hidden_states = encoder_outputs[0]
        decoder_inputs_ids = self._shift_right(labels)

        code_len = decoder_inputs_ids.shape[-1]
        decoder_inputs_embeds = []
        for i in range(code_len):
            code_embedding = self.cid_embed_list[i]
            decoder_inputs_embeds.append(code_embedding(decoder_inputs_ids[:,i]))

        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds,dim=1)

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        assert code_len == sequence_output.shape[1]
        lm_logits = []
        for i in range(code_len):
            out_head = self.centroids[i]
            lm_logits.append(out_head(sequence_output[:,i])) # [b,codebook_size]

        lm_logits = torch.stack(lm_logits,dim=1) # [b,code_len,codebook_size]
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            decoder_hidden_states=decoder_outputs.last_hidden_state,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



    def forward_train(self,
                      content_input_ids: Optional[torch.LongTensor] = None,
                      atom_input_ids: Optional[torch.LongTensor] = None,
                      atom_index: Optional[torch.LongTensor] = None,
                      attention_mask: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      user_atom_ids: Optional[torch.LongTensor] = None,  # 当ty是item的时候使用
                      user_atom_index: Optional[torch.LongTensor] = None,  # 当ty是item的时候使用
                      input_positions: Optional[torch.LongTensor] = None,
                      output_attentions: Optional[bool] = None,
                      output_hidden_states: Optional[bool] = None,
                      use_cache: Optional[bool] = None,
                      return_dict: Optional[bool] = None,
                      ty='user'):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = atom_input_ids.shape[0]
        device = atom_input_ids.device
        prompt_embed = self.get_prompt_embed(batch_size,device,ty)

        assert atom_index.shape == atom_input_ids.shape
        contend_embed = self.shared(content_input_ids)
        atomid_embed = self.atomid_embed(atom_input_ids)
        embed_shape = atomid_embed.shape[-1]
        exindex = atom_index.unsqueeze(-1).expand(*atom_index.shape,embed_shape)
        encoder_input_embed = contend_embed.scatter(dim=1,index=exindex,src=atomid_embed)
        encoder_input_embed[:, 0] = prompt_embed

        if ty == 'item' and user_atom_ids is not None:
            assert user_atom_ids.shape == user_atom_index.shape
            user_atomid_embed = self.atomid_user_embed(user_atom_ids)
            user_exindex = user_atom_index.unsqueeze(-1).expand(*user_atom_index.shape,embed_shape)
            encoder_input_embed = encoder_input_embed.scatter(dim=1,index=user_exindex,src=user_atomid_embed)

        if self.info_position is not None and input_positions is not None:
            assert input_positions.shape == encoder_input_embed.shape[:2]
            input_position_embed = self.info_position(input_positions)
            encoder_input_embed += input_position_embed

        # 2. encoder过
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_input_embed,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        decoder_inputs_ids = self._shift_right(labels)

        code_len = decoder_inputs_ids.shape[-1] # 3
        # print(code_len)
        decoder_inputs_embeds = []
        for i in range(code_len):
            code_embedding = self.cid_embed_list[i]
            decoder_inputs_embeds.append(code_embedding(decoder_inputs_ids[:,i]))

        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds,dim=1)

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)


        assert code_len == sequence_output.shape[1]
        lm_logits = []
        for i in range(code_len):
            out_head = self.centroids[i]
            lm_logits.append(out_head(sequence_output[:,i])) # [b,codebook_size]

        lm_logits = torch.stack(lm_logits,dim=1) # [b,code_len,codebook_size]
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            decoder_hidden_states=decoder_outputs.last_hidden_state,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def predict(self, generative_config,
                content_input_ids: Optional[torch.LongTensor] = None,
                atom_input_ids: Optional[torch.LongTensor] = None,
                atom_index: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                input_positions: Optional[torch.FloatTensor] = None,

                pretype='user'):
        batch_size = atom_input_ids.shape[0]
        device = atom_input_ids.device

        prompt_embed = self.get_prompt_embed(batch_size,device,pretype)

        assert atom_index.shape == atom_input_ids.shape
        contend_embed = self.shared(content_input_ids)
        atomid_embed = self.atomid_embed(atom_input_ids)
        embed_shape = atomid_embed.shape[-1]

        exindex = atom_index.unsqueeze(-1).expand(*atom_index.shape, embed_shape)

        encoder_input_embed = contend_embed.scatter(dim=1, index=exindex, src=atomid_embed)
        encoder_input_embed[:,0] = prompt_embed

        if self.info_position is not None and input_positions is not None:
            assert input_positions.shape == encoder_input_embed.shape[:2]
            input_position_embed = self.info_position(input_positions)
            encoder_input_embed += input_position_embed

        attention_mask = attention_mask
        output = self.generate(inputs_embeds=encoder_input_embed,attention_mask=attention_mask,**generative_config)

        return output

    def get_prompt_embed(self,batch_size,device,ty='user'):
        assert ty in ['user','item'],'the predict type must be in [user,item]'
        if ty == 'user':
            prompt = torch.ones(batch_size,dtype=torch.long).to(device)
        else:
            prompt = torch.zeros(batch_size,dtype=torch.long).to(device)

        prompt_embed = self.prompt(prompt)

        return prompt_embed

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        row=1,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)


        code_index = 0
        if past_key_values is not None:
            code_index = past_key_values[0][0].shape[2]
        code_embedding = self.cid_embed_list[code_index]
        decoder_inputs_embeds = code_embedding(decoder_input_ids) #[b,1,d_model]

        decoder_outputs = self.decoder(
            # input_ids=decoder_input_ids, 
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        out_head = self.centroids[code_index]
        lm_logits = out_head(sequence_output)
        lm_logits = lm_logits*row

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



    # NOTE: t5
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        row=1,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "row":row
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            # logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            print("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past