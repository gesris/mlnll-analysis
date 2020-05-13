import argparse
import os
import pickle

from ntuple_processor import RunManager

import logging
logger = logging.getLogger('')


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main(args):
    graphs = pickle.load(open(os.path.join(args.workdir, 'graphs.pickle'), 'rb'))

    arguments = []
    for i in range(len(graphs)):
        arguments.append([i, os.getcwd(), args.workdir])

    jobdir = os.path.join(args.workdir, 'shapes_jobs')
    logger.info('Write job files for %s jobs to %s', len(arguments), jobdir)

    with open(os.path.join(jobdir, 'arguments.txt'), 'w') as f:
        for jobargs in arguments:
            f.write('{} {} {}\n'.format(jobargs[0], jobargs[1], jobargs[2]))

    with open(os.path.join(jobdir, 'job.jdl'), 'w') as f:
        f.write('''\
universe = docker
docker_image = mschnepf/slc7-condocker
executable = {}
output = out/$(cluster).$(Process).out
error = err/$(cluster).$(Process).err
log = log/$(cluster).$(Process).log
Requirements = ( (Target.ProvidesIO == False) && (TARGET.ProvidesEKPResources == True) )
+RequestWalltime = 1800
+ExperimentalJob = True
RequestMemory = 2000
RequestCpus = 1
max_retries = 5
accounting_group = cms.higgs
queue arguments from arguments.txt
'''.format(os.path.join(os.getcwd(), 'shapes/job.sh')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'shapes.log'), logging.INFO)
    main(args)
