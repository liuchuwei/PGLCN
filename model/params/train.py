import argparse
from argparse import ArgumentDefaultsHelpFormatter
from utils.io_utils import obtain_dataset
from utils.model_utils import obtain_placeholders, build_model
from utils.parser_utils import set_defaults_train
from utils.train_utils import train_model, set_seed

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    # project
    parser.add_argument('--project', dest='project',
                           help='Possible values:'
                                'Diff_PC: stad_PC1, stad_PC2...'
                                'train_stad_2pc_5exp...')

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
                        help='Method. Possible values: base, gat, glcn, sglcn'
                             'decision_tree, sgd, random_forest, adaboost,'
                             'svc_linear, svc_rbf,')
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

    ## gcn
    parser.add_argument('--input_dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num_classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--train_ratio', dest='train_ratio', type=int,
                        help='Ratio of sample for training')
    parser.add_argument('--test_ratio', dest='test_ratio', type=int,
                        help='Ratio of sample for testing')

    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    parser.add_argument('--bn', dest='bn', action='store_const',
                        const=True, default=False,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')

    return parser

def main(args):
    # set default parser
    set_defaults_train(args)

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
