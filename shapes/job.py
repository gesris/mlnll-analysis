import argparse
import os
import pickle

from ntuple_processor import RunManager


def main(args):
    print('Load graphs')
    graphs = pickle.load(open(os.path.join(args.workdir, 'graphs.pickle'), 'rb'))
    print('Select graph with base node:\n{}'.format(graphs[args.jobid]))

    print('Initialize run manager')
    r_manager = RunManager([graphs[args.jobid]])

    print('Run job')
    r_manager.run_locally(
            os.path.join(args.workdir, 'shapes_files', 'shapes_{}.root'.format(args.jobid)),
            nworkers=1, nthreads=1)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('jobid', type=int, help='Job ID corresponding to index of the graph')
    args = parser.parse_args()
    main(args)
