import random

from train.TrainPglcn import *
from train.TrainML import *
from train.TrainSglcn import *
from train.TrainNodeClassifier import *

def train_model(model, args, log=None, dataset=None):

    if args.method == "pglcn":
        if args.project=="pretrain_stad":
            train_pglcn(model, args=args, dataset=dataset)
        else:
            train_pglcn_iteration(model, args=args, dataset=dataset)

    if args.method == "sglcn":
        train_sglcn(model, args=args)

    if args.dataset == "syn1" and args.method == "gcn":
        G, labels, name = dataset
        train_node_classifier(G, labels, model, args=args, writer=log)

    if args.dataset in ["syn1", "syn2", "syn3", "syn4", "syn5"]  and args.method == "glcn" :
        G, labels, name = dataset
        train_node_glcn_classifier(G, labels, model, args=args, writer=log)

    # if args.method == "pglcn":
    #     train_pglcn(model, args=args, log=None, dataset=dataset)

    if args.method == "sglcn":
        train_sglcn(model, args=args)

    if args.dataset in ["syn1", "syn2", "syn3", "syn4", "syn5"]  and args.method == "gcn":
        G, labels, name = dataset
        train_node_classifier(G, labels, model, args=args, writer=log)

    if args.method in ["sgd", "svc_rbf", "svc_linear", "random_forest",
                             "adaboost", "decision_tree"] :

        train_ML(model, args=args, dataset=dataset)



def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import torch.optim as optim

def build_optimizer(args, params, weight_decay=0.0):
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer