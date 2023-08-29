import argparse
from argparse import ArgumentDefaultsHelpFormatter


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    return parser

def main(args):
    pass