import argparse

def cifar_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", default = 0, help="Utilize which gpu", type = int)
    parser.add_argument("--nbatch", default = 128, help="size of batch", type = int)

    return parser

def imgnet_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", default = 0, help="Utilize which gpu", type = int)

    return parser

def car_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default = 0, help="Utilize which gpu", type = int)
    return parser

