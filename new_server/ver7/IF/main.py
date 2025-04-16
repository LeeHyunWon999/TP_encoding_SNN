import os
import json
import argparse

from util import util


def main():

    # CLI 통합 동작...이지만 당장 여기선 해당 기능 확장 안함
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) 
    args.update(param)

    util.execute(args)


def load_json(path):
    with open(path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser   = argparse.ArgumentParser(description='SNN encoding')
    
    parser.add_argument('config')
   
    return parser


if __name__ =="__main__":

    main()
    