import torch
from profiler import profileit
import argparse

parser = argparse.ArgumentParser(description='Vector DB')
parser.add_argument('-p', '--profile', dest='profile', action='store_true',
                    default=False,
                    help='to profile functions in the app')
args = parser.parse_args()


@profileit(enabled=args.profile)
def main():
    print("not sure what to implement!")

main()