import os

import numpy as np
import torch

from utils import gengraph, featgen
# dataset: stad, coad, ucec, syn1, syn2, syn3, syn4, syn5, citeseer, cora
def obtain_dataset(args):

    if args.dataset == "syn1":
        G, labels, name = gengraph.gen_syn1(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset == "syn2":
        G, labels, name = gengraph.gen_syn2()
        return G, labels, name

    if args.dataset == "syn3":
        G, labels, name = gengraph.gen_syn3(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset == "syn4":
        G, labels, name = gengraph.gen_syn4(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset == "syn5":
        G, labels, name = gengraph.gen_syn5(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset in ["stad", "coad", "ucec"]:
        adj, features, second, masks_list, labels, Reac_feat = gengraph.load_pathway_graph(args)
        return adj, features, second, masks_list, labels, Reac_feat

    if args.dataset == "cora":
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = gengraph.load_cite_data(args)
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    if args.dataset == "citeseer":
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = gengraph.load_cite_data(args)
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def gen_prefix(args):
    '''Generate label prefix for a graph model.
    '''
    # if args.bmname is not None:
    #     name = args.bmname
    # else:
    name = args.dataset
    name += "_" + args.method

    # name += "_h" + str(args.hidden_dim) + "_o" + str(args.output_dim)
    # if not args.bias:
    #     name += "_nobias"
    # if len(args.name_suffix) > 0:
    #     name += "_" + args.name_suffix
    return name



def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    filename = os.path.join(save_dir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + ".pth.tar"


def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.

    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """

    filename = args.dataset + "_" + args.method + "_pretrain_for_explain." + "pth.tar"
    # create parent directory
    par_dir = "log/" + args.dataset + "_explain"

    if not os.path.exists(par_dir):
        os.makedirs(par_dir)

    filename = par_dir + "/" + filename
    if isinstance(optimizer, list):
        torch.save(
            {
                "epoch": num_epochs,
                "model_type": args.method,
                "optimizer": optimizer,
                "model_state": model.state_dict(),
                "optimizer_state": [i.state_dict() for i in optimizer],
                "cg": cg_dict,
            },
            filename,
        )

    else:
        torch.save(
            {
                "epoch": num_epochs,
                "model_type": args.method,
                "optimizer": optimizer,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cg": cg_dict,
            },
            filename,
        )

