import argparse

def cifar_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbatch", default = 128, help="size of batch", type = int)
    return parser
