import argparse
from argparse import ArgumentDefaultsHelpFormatter

# from utils.parser_utils import set_defaults
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

    return parser

def main(args):

    # set default parser
    set_defaults(args)

    # set seed
    set_seed(args.seed)