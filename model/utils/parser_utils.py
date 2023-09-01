def set_defaults_explain(args):
    pass

def set_defaults_train(args):

    if args.project in ["syn1_glcn", "syn2_glcn", "syn3_glcn", "syn4_glcn", "syn5_glcn"]:

        # set seed
        args.seed = 666666

        # dataset
        args.dataset = args.project.split("_")[0]

        args. method = 'glcn'
        args.opt_scheduler = 'none'
        args.placeholders = False
        args.dropout = 0.0
        args.weight_decay = 0.005
        args.input_dim = 10
        args.hidden_dim = 20
        args.output_dim = 20
        args.num_classes = 2
        args.num_gc_layers = 3
        args.gpu = True
        args.clip = 2.0
        args.batch_size = 20
        args.epoch = 1000
        args.train_ratio = 0.8
        args.test_ratio = 0.1
        if not args.dataset:
            args.losslr1 = 1e2
            args.losslr2 = 0.0001
        else:
            args.losslr1 = 1e-6    # 1e-6
            args.losslr2 = 0.0001 # 0.0001
            args.epoch = 1000

        args.lr1 = 0.001
        args.lr2 = 0.001

    if args.project in ["syn1_gcn", "syn2_gcn", "syn3_gcn", "syn4_gcn", "syn5_gcn"]:

        # set seed
        args.seed = 666666

        # dataset
        args.dataset = args.project.split("_")[0]

        # model
        args.method = 'gcn'
        args.opt_scheduler = 'none'
        args.placeholders = False
        args.dropout = 0.0
        args.weight_decay = 0.005
        args.input_dim = 10
        args.hidden_dim = 20
        args.output_dim = 20
        args.num_classes = 2
        args.num_gc_layers = 3
        args.gpu = True
        args.lr = 0.001
        args.clip = 2.0
        args.batch_size = 20
        args.epoch = 1000
        args.train_ratio = 0.8
        args.test_ratio = 0.1

        return args

    if args.project == "stad_pglcn":
        pass

    if args.project in ['train_stad_2pc_5exp', 'train_coad_2pc_5exp', 'train_ucec_2pc_5exp',
                        'pretrain_stad']:

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
            args.dropout2=0.0  # graph gcn dropout 0.
            args.dropout3=0.0  # dense dropout  0.
            args.bias=True
            args.weight_decay=1e-4  # 1e-4
            args.hidden_gl=70 # 70
            args.hidden_gcn=25 # 25

            # train
            args.seed=666666
            args.iexp=5
            args.lr1=0.1
            args.lr2=0.00005   # 0.00005
            args.losslr1=0.01
            args.losslr2=0.00001
            args.epoch=100
            args.clip=2.0
            args.early_stopping=10

        if args.dataset == "coad":
            args.placeholders = True
            args.dropout1 = 0.6  # graph learn dropout 0.6
            args.dropout2 = 0.0  # graph gcn dropout 0.
            args.dropout3 = 0.0  # dense dropout  0.
            args.bias = True
            args.weight_decay = 1e-4  # 1e-4
            args.hidden_gl = 70  # 70
            args.hidden_gcn = 30  # 30

            # train
            args.seed = 666666
            args.iexp = 5
            args.lr1 = 0.1
            args.lr2 = 0.00005  # 0.00005
            args.losslr1 = 0.01
            args.losslr2 = 0.00001
            args.epoch = 100
            args.clip = 2.0
            args.early_stopping = 10

        if args.dataset == "ucec":
            args.placeholders = True
            args.dropout1 = 0.6  # graph learn dropout 0.3
            args.dropout2 = 0.  # graph gcn dropout 0
            args.dropout3 = 0.  # dense dropout  0
            args.bias = True
            args.weight_decay = 1e-4  # 1e-4
            args.hidden_gl = 70  # 70
            args.hidden_gcn = 20  # 20

            # train
            args.seed = 666666
            args.iexp = 5
            args.lr1 = 0.1
            args.lr2 = 0.00005  # 0.00005
            args.losslr1 = 0.01
            args.losslr2 = 0.00001
            args.epoch = 100
            args.clip = 2.0
            args.early_stopping = 10

    if args.project in ["stad_PC1", "stad_PC3", "stad_PC4", "stad_PC5",
           "coad_PC1", "coad_PC3", "coad_PC4", "coad_PC5",
           "ucec_PC1", "ucec_PC3", "ucec_PC4", "ucec_PC5",
           "stad_g", "stad_c", "stad_m","stad_gc", "stad_gm", "stad_cm",
           "coad_g", "coad_c", "coad_m", "coad_gc", "coad_gm", "coad_cm",
           "ucec_g", "ucec_c", "ucec_m", "ucec_gc", "ucec_gm", "ucec_cm"
           ]:

        args.seed = 666666
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

        if args.dataset == "stad":
            args.placeholders = True
            args.dropout1 = 0.6  # graph learn dropout 0.6
            args.dropout2 = 0.0  # graph gcn dropout 0.
            args.dropout3 = 0.0  # dense dropout  0.
            args.bias = True
            args.weight_decay = 1e-4  # 1e-4
            args.hidden_gl = 70  # 70
            args.hidden_gcn = 25  # 25

            # train
            args.seed = 666666
            args.iexp = 5
            args.lr1 = 0.1
            args.lr2 = 0.00005  # 0.00005
            args.losslr1 = 0.01
            args.losslr2 = 0.00001
            args.epoch = 100
            args.clip = 2.0
            args.early_stopping = 10

        if args.dataset == "coad":
            args.placeholders = True
            args.dropout1 = 0.6  # graph learn dropout 0.6
            args.dropout2 = 0.0  # graph gcn dropout 0.
            args.dropout3 = 0.0  # dense dropout  0.
            args.bias = True
            args.weight_decay = 1e-4  # 1e-4
            args.hidden_gl = 70  # 70
            args.hidden_gcn = 30  # 30

            # train
            args.seed = 666666
            args.iexp = 5
            args.lr1 = 0.1
            args.lr2 = 0.00005  # 0.00005
            args.losslr1 = 0.01
            args.losslr2 = 0.00001
            args.epoch = 100
            args.clip = 2.0
            args.early_stopping = 10

        if args.dataset == "ucec":
            args.placeholders = True
            args.dropout1 = 0.6  # graph learn dropout 0.3
            args.dropout2 = 0.  # graph gcn dropout 0
            args.dropout3 = 0.  # dense dropout  0
            args.bias = True
            args.weight_decay = 1e-4  # 1e-4
            args.hidden_gl = 70  # 70
            args.hidden_gcn = 20  # 20

            # train
            args.seed = 666666
            args.iexp = 5
            args.lr1 = 0.1
            args.lr2 = 0.00005  # 0.00005
            args.losslr1 = 0.01
            args.losslr2 = 0.00001
            args.epoch = 100
            args.clip = 2.0
            args.early_stopping = 10

    if args.method in ['decision_tree', 'sgd', 'random_forest', 'adaboost','svc_linear', 'svc_rbf'
           ]:
        args.item="PC2"
        args.npc=2
        args.omic=3
        args.iexp=5
        args.imbalance=True
        args.seed = 666666

    return args