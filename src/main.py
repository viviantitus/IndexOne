from profiler import profileit
import argparse
import kmeans

parser = argparse.ArgumentParser(description='Vector DB')
parser.add_argument('-p', '--profile', dest='profile', action='store_true',
                    default=False,
                    help='to profile functions in the app')
args = parser.parse_args()


@profileit(enabled=args.profile)
def main():
    dataset = kmeans.get_dataset(10000)
    centroids, assignments = kmeans.compute_kmeans(dataset, 4, 1000)
    print(assignments)

main()