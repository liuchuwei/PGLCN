import argparse
import os.path
from argparse import ArgumentDefaultsHelpFormatter

import numpy as np

from utils import io_utils
from utils.model_utils import build_model, build_explainer
from utils.parser_utils import set_defaults_explain
from utils.train_utils import set_seed


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    # project
    parser.add_argument('--project', dest='project',
                           help='Possible values:'
                                'Dataset_method: stad_pglcn...')

    # set seed
    parser.add_argument('--seed', dest='seed', type=float,
                            help='Seed.')
    # dataset & method
    parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset. Possible values: '
                                'stad, syn1, syn2, syn3, syn4, syn5')
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: gcn, glcn')

    # io utils
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--logdir", dest="logdir", help="Log directory")
    parser.add_argument("--explainer_suffix", dest="explainer_suffix", help="Suffix of explainer")

    # build model
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=True,
        help="whether to use GPU.",
    )

    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )

    # pglcn
    parser.add_argument('--hidden_gl', dest='hidden_gl', type=int,
                        help='Hidden gl dimension')
    parser.add_argument('--hidden_gcn', dest='hidden_gcn', type=int,
                        help='Hidden gcn dimension')
    parser.add_argument('--dropout1', dest='dropout', type=int,
                        help='Graph learn dropout ratio')
    parser.add_argument('--dropout2', dest='dropout', type=int,
                        help='Graph gcn dropout ratio')
    parser.add_argument('--dropout3', dest='dropout', type=int,
                        help='Dense dropout ratio')
    parser.add_argument('--weight_decay', dest='weight_decay', type=int,
                        help='Weight_decay ratio')
    parser.add_argument('--Placeholders', dest='placeholders', type=bool,
                        help='Placeholders')
    parser.add_argument('--Bias', dest='bias', type=bool,
                        help='bias')
    parser.add_argument('--lr1', dest='lr1', type=float,
                            help='Sparse learning rate.')
    parser.add_argument('--lr2', dest='lr2', type=float,
                            help='Ce learning rate.')
    parser.add_argument('--losslr1', dest='losslr1', type=float,
                            help='Sparse loss weight.')
    parser.add_argument('--losslr2', dest='losslr2', type=float,
                            help='Ce loss weight.')
    parser.add_argument('--omic', dest='omic',
                           help='Number of omic.')
    parser.add_argument('--npc', dest='npc',
                           help='Number of pca component.')

    # explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--opt", dest="opt", type=str, help="Optimizer."
    )
    parser.add_argument('--lr', dest='lr', type=float,
                            help='Learining rate.')
    parser.add_argument(
        "--num_epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, help="Batch size."
    )
    parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')

    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')

    return parser

def main(args):

    # set default parser
    set_defaults_explain(args)

    # set seed
    set_seed(args.seed)

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(args)
    cg_dict = ckpt["cg"] # get computation graph

    # build model
    if args.method == "pglcn":
        placeholder = cg_dict["placeholder"]
        model = build_model(args=args, placeholders=placeholder)
    else:
        model = build_model(args=args, dataset=cg_dict)

    # load state_dict
    if args.gpu:
        model = model.cuda()
    model.load_state_dict(ckpt["model_state"])

    # Create explainer
    explainer = build_explainer(cg_dict, model, args)

    # Explain
    if args.method == "pglcn":

        if not os.path.exists("log/stad_explain/feat_mask.npy"):
            masked_adj = explainer.explain_path_stats(
                0, args
            )
            adj_mask = [item[2] for item in masked_adj]
            adj_mask = np.array(adj_mask)

            io_utils.save_stad_Mask(masked_adj, args)
        else:
            adj_mask = io_utils.loadMask(args)

        # visualization
        if args.combine_imm:
            io_utils.DrawPathMask(adj_mask, args, cg_dict, imm=True, show_label=False)
        else:
            io_utils.DrawPathMask(adj_mask, args, cg_dict)

    elif args.dataset == "syn1":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(400, 700, 5), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [300], args
        )

    elif args.dataset == "syn2":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(400, 700, 5), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [350], args
        )

    elif args.dataset == "syn3":


        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(301, 1021, 9), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [301], args
        )

    elif args.dataset == "syn4":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(511, 871, 6), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [511], args
        )

    elif args.dataset == "syn5":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(512, 1232, 9), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [548], args
        )
