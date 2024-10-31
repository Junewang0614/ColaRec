
v4T5small_train = dict(
    seed=0,
    # batch_size=64,
    batch_size=128,
    eval_batch_size=16,
    num_workers=4,
    n_epochs=80,
    total_lr=5e-4,
    lr_1=1e-3,
    eval_steps=5,
    print_steps=500,
    early_stop=10,
    generate_lamda=0.08,
    content_cl_lamda = 0.05,
    topk=[5, 10,20],
    metrics=['recall', 'NDCG'],
    key_metric='NDCG@10',

    title=True,

    num_beams=100,
    num_return_sequences=100,
    row=1,

    version=1,
    info='',
    item_position = False,
    show_process = True,
    seq_type = 'short'
)
