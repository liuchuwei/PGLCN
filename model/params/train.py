import argparse
from argparse import ArgumentDefaultsHelpFormatter
from utils.io_utils import obtain_dataset
from utils.model_utils import obtain_placeholders, build_model
from utils.parser_utils import set_defaults
from utils.train_utils import train_model, set_seed

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    # project
    parser.add_argument('--project', dest='project',
                           help='Possible values:'
                                'train_stad_2pc_30exp')

    # dataset
    parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset. Possible values: '
                                'stad, coad, ucec, syn1, syn2, syn3, syn4, syn5, citeseer, cora')
    parser.add_argument('--item', dest='item',
                           help='Item of input dataset.')
    parser.add_argument('--omic', dest='omic',
                           help='Number of omic.')
    parser.add_argument('--npc', dest='npc',
                           help='Number of pca component.')
    parser.add_argument('--iexp', dest='iexp',
                           help='Iterations of experiment.')
    parser.add_argument('--imbalance', dest='imbalance',
                           help='Whether imbalance the dataset.')

    # device
    parser.add_argument('--gpu', dest='gpu', type=bool,
                        help='Whether to use gpu')

    # model
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, gat, glcn, simaese_gcn, sglcn'
                             'decision tree, l2 logistic regression, random forest, adaptive boosting,'
                             'linear support vector machine, RBF support vector machine,')
    ## pglcn
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

    # train
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--epoch', dest='epoch', type=int,
                            help='Training epoch.')
    opt_parser.add_argument('--early_stopping', dest='early_stopping', type=int,
                            help='early_stopping steps')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--seed', dest='seed', type=float,
                            help='Seed.')

    ## pglcn
    opt_parser.add_argument('--lr1', dest='lr1', type=float,
                            help='Sparse learning rate.')
    opt_parser.add_argument('--lr2', dest='lr2', type=float,
                            help='Ce learning rate.')
    opt_parser.add_argument('--losslr1', dest='losslr1', type=float,
                            help='Sparse loss weight.')
    opt_parser.add_argument('--losslr2', dest='losslr2', type=float,
                            help='Ce loss weight.')

    return parser

def main(args):
    # set default parser
    set_defaults(args)

    # set seed
    set_seed(args.seed)

    # obtain dataset
    dataset = obtain_dataset(args)

    # build model
    if args.placeholders:
        placeholders = obtain_placeholders(args, dataset)     # placeholders
        model = build_model(args=args, placeholders=placeholders)
    else:
        model = build_model(args=args, dataset=dataset)

    # train model
    train_model(model=model, args=args, dataset=dataset)
