import argparse
import os
import pickle

from ntuple_processor import RunManager


def job(workdir, jobid):
    print('Load graphs')
    graphs = pickle.load(open(os.path.join(workdir, 'graphs.pickle'), 'rb'))
    print('Select graph with base node:\n{}'.format(graphs[jobid]))

    print('Initialize run manager')
    r_manager = RunManager([graphs[jobid]])

    print('Run job')
    r_manager.run_locally(
            os.path.join(workdir, 'shapes_files', 'shapes_{}.root'.format(jobid)),
            nworkers=1, nthreads=1)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('jobid', type=int, help='Job ID corresponding to index of the graph')
    job(args.workdir, args.jobid)
