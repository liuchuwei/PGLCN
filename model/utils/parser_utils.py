def set_defaults(args):

    if args.project == 'train_stad_2pc_30exp':

        # dataset
        args.dataset='stad'
        args.method='pglcn'
        args.item="PC2"
        args.omic=3
        args.npc=2
        args.imbalance=True

        # device
        args.gpu=True

        # model
        args.placeholders=True
        args.dropout1=0  # graph dropout
        args.dropout2=0  # dense dropout
        args.bias=True
        args.weight_decay=1e-8
        args.hidden_gl=70
        args.hidden_gcn=30

        # train
        args.iexp=30
        args.lr1=0.1
        args.lr2=0.00005
        args.losslr1=0.01
        args.losslr2=0.00001
        args.epoch=1000
        args.clip=2.0
        args.early_stopping=100

    return args