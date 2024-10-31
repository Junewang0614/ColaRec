# training
python model_train.py \
  --info="phone-short-model-gen008-cl01" \
  --seq_type="short" --dataset="phone" --cid_token_num=32 \
  --batch_size=128 --generate_lamda=0.08 --content_cl_lamda=0.1

# evaling
# --eval_model_mkdir 训练保存的模型路径
python model_eval.py \
--eval_model_mkdir="" \
--dataset="phone" --seq_type="short" --n_epochs=100 --start=1 --eval_batch_size=20
