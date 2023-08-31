def set_defaults(args):

    if args.project in ['train_stad_2pc_30exp', 'train_coad_2pc_30exp', 'train_ucec_2pc_30exp']:

        # dataset
        args.dataset=args.project.split("_")[1]
        args.method='pglcn'
        args.item="PC2"
        args.omic=3
        args.npc=2
        args.imbalance=True

        # device
        args.gpu=True

        # model
        if args.dataset=="stad":
            args.placeholders=True
            args.dropout1=0.6  # graph learn dropout 0.6
            args.dropout2=0.  # graph gcn dropout 0
            args.dropout3=0.  # dense dropout  0
            args.bias=True
            args.weight_decay=1e-8  # 1e-8
            args.hidden_gl=70 # 70
            args.hidden_gcn=30 # 30

            # train
            args.seed=666666
            args.iexp=30
            args.lr1=0.1
            args.lr2=0.00005   # 0.00005
            args.losslr1=0.01
            args.losslr2=0.00001
            args.epoch=300
            args.clip=2.0
            args.early_stopping=10

        if args.dataset == "coad":
            args.placeholders = True
            args.dropout1 = 0.6  # graph learn dropout 0.6
            args.dropout2 = 0.3  # graph gcn dropout 0.3
            args.dropout3 = 0.3  # dense dropout  0.3
            args.bias = True
            args.weight_decay = 1e-4  # 1e-4
            args.hidden_gl = 70  # 70
            args.hidden_gcn = 30  # 30

            # train
            args.seed = 666666
            args.iexp = 30
            args.lr1 = 0.1
            args.lr2 = 0.00005  # 0.00005
            args.losslr1 = 0.01
            args.losslr2 = 0.00001
            args.epoch = 300
            args.clip = 2.0
            args.early_stopping = 10

        if args.dataset == "ucec":
            args.placeholders = True
            args.dropout1 = 0.6  # graph learn dropout 0.3
            args.dropout2 = 0.0  # graph gcn dropout 0
            args.dropout3 = 0.0  # dense dropout  0
            args.bias = True
            args.weight_decay = 1e-8  # 1e-8
            args.hidden_gl = 90  # 90
            args.hidden_gcn = 25  # 25

            # train
            args.seed = 666666
            args.iexp = 30
            args.lr1 = 0.1
            args.lr2 = 0.00005  # 0.00005
            args.losslr1 = 0.01
            args.losslr2 = 0.00001
            args.epoch = 300
            args.clip = 2.0
            args.early_stopping = 10

    if args.project in ["stad_PC1", "stad_PC2", "stad_PC3", "stad_PC4", "stad_PC5",
           "coad_PC1", "coad_PC2", "coad_PC3", "coad_PC4", "coad_PC5",
           "ucec_PC1", "ucec_PC2", "ucec_PC3", "ucec_PC4", "ucec_PC5",
           "stad_g", "stad_c", "stad_m","stad_gc", "stad_gm", "stad_cm",
           "coad_g", "coad_c", "coad_m", "coad_gc", "coad_gm", "coad_cm",
           "ucec_g", "ucec_c", "ucec_m", "ucec_gc", "ucec_gm", "ucec_cm"
           ]:

        # dataset
        args.dataset = args.project.split("_")[0]
        args.method = 'pglcn'
        args.item = args.project.split("_")[1]

        if "PC" in args.project:
            args.omic = 3
            args.npc = int(args.project.split("PC")[1])
        else:
            args.omic = len(args.project.split("_")[1])
            args.npc = 2

        args.imbalance = True

        # device
        args.gpu = True

        # model
        args.placeholders = True
        args.dropout1 = 0.6  # graph learn dropout 0.6
        args.dropout2 = 0.  # graph gcn dropout 0.3; 0
        args.dropout3 = 0.  # dense dropout 0.3; 0
        args.bias = True
        args.weight_decay = 1e-8  # 1e-8
        args.hidden_gl = 70  # 70
        args.hidden_gcn = 30  # 30

        # train
        args.seed = 666666
        args.iexp = 1
        args.lr1 = 0.1
        args.lr2 = 0.00005  # 0.00005
        args.losslr1 = 0.01
        args.losslr2 = 0.00001
        args.epoch = 300
        args.clip = 2.0
        args.early_stopping = 10

    if args.method in ['decision_tree', 'sgd', 'random_forest', 'adaboost','svc_linear', 'svc_rbf'
           ]:
        args.item="PC2"
        args.npc=2
        args.omic=3
        args.iexp=30
        args.imbalance=True
        args.seed = 666666

    return args