import os
import argparse
import pickle

import ROOT
ROOT.DisableImplicitMT() # Otherwise the friends would be not ordered
import numpy as np
np.random.seed(1234)
from utils import config as cfg

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1234)

from array import array

from collect_events import make_dataset
from train import model

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


foldernames = [
        'mt_nominal',
        'mt_jerUncUp',
        'mt_jerUncDown',
        'mt_jecUncAbsoluteYearUp',
        'mt_jecUncAbsoluteYearDown',
        'mt_jecUncAbsoluteUp',
        'mt_jecUncAbsoluteDown',
        'mt_jecUncBBEC1YearUp',
        'mt_jecUncBBEC1YearDown',
        'mt_jecUncBBEC1Up',
        'mt_jecUncBBEC1Down',
        'mt_jecUncEC2YearUp',
        'mt_jecUncEC2YearDown',
        'mt_jecUncEC2Up',
        'mt_jecUncEC2Down',
        'mt_jecUncHFYearUp',
        'mt_jecUncHFYearDown',
        'mt_jecUncHFUp',
        'mt_jecUncHFDown',
        'mt_jecUncFlavorQCDUp',
        'mt_jecUncFlavorQCDDown',
        'mt_jecUncRelativeSampleYearUp',
        'mt_jecUncRelativeSampleYearDown',
        'mt_jecUncRelativeBalUp',
        'mt_jecUncRelativeBalDown',
        'mt_tauEsThreeProngUp',
        'mt_tauEsThreeProngDown',
        'mt_tauEsThreeProngOnePiZeroUp',
        'mt_tauEsThreeProngOnePiZeroDown',
        'mt_tauEsOneProngUp',
        'mt_tauEsOneProngDown',
        'mt_tauEsOneProngOnePiZeroUp',
        'mt_tauEsOneProngOnePiZeroDown',
        ]


def main(args):
    modelpath_fold0 = os.path.join(args.workdir, 'model_fold0.h5')
    modelpath_fold1 = os.path.join(args.workdir, 'model_fold1.h5')
    arguments = []
    for process in cfg.files: #[0:10] für schnellere effizienz! später löschen
        for filename in cfg.files[process]:
            for folder in foldernames:
                # Check whether the input file and folder exist
                # Just skip over missing folders but break for missing files
                filepath = os.path.join(cfg.ntuples_base, filename, filename + '.root')
                folderpath = folder + '/ntuple'
                logger.debug('Collect arguments for folder {} of file {}'.format(folderpath, filepath))
                f = ROOT.TFile(filepath, 'READ')
                if f == None:
                    logger.fatal('File {} does not exist'.format(filepath))
                    raise Exception

                dir_ = f.Get(folderpath)
                if dir_ == None:
                    logger.debug('Skipping over folder {} in file {} (folder does not exist)'.format(folderpath, filepath))
                    f.Close()
                    continue
                f.Close()

                # Try to make output folder if not yet existent
                try:
                    os.mkdir(os.path.join(args.workdir, 'MLScores', filename))
                except:
                    pass

                # Add to argument list
                arguments.append([args.workdir, folder, filename])


    # Write argument list to jobdir
    jobdir = os.path.join(args.workdir, 'MLScores_jobs')
    with open(os.path.join(jobdir, 'arguments.txt'), 'w') as f:
        for i, a in enumerate(arguments):
            logger.debug('Job {}: {}, {}, {}'.format(i, a[0], a[1], a[2]))
            f.write('{} {} {} {} {}\n'.format(i, os.getcwd(), a[0], a[1], a[2]))

    # Write JDL to jobdir
    with open(os.path.join(jobdir, 'job.jdl'), 'w') as f:
        f.write('''\
universe = docker
docker_image = mschnepf/slc7-condocker
executable = {}
output = out/$(cluster).$(Process).out
error = err/$(cluster).$(Process).err
log = log/$(cluster).$(Process).log
Requirements = ( (TARGET.Cloudsite =!= "blade") && (TARGET.ProvidesEKPResources == True) )
+RequestWalltime = 1800
RequestMemory = 2000
RequestCpus = 1
max_retries = 5
accounting_group = cms.higgs
queue arguments from arguments.txt
'''.format(os.path.join(os.getcwd(), 'ml/job.sh')))

    # Test a single job
    application(*arguments[-1]) # wichtig fürs testen


def application(workdir, folder, filename):
    print('Start application')

    # Create session
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session = tf.Session(config=config)

    # Load models
    def load_model(x, fold):
        _, f = model(x, len(cfg.ml_variables), fold)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_fold{}'.format(fold))
        path = tf.train.latest_checkpoint(os.path.join(workdir, 'model_fold{}'.format(fold)))
        print('Load variables for fold {} from {}'.format(fold, path))
        saver = tf.train.Saver(variables)
        saver.restore(session, path)
        return f

    x_ph = tf.placeholder(tf.float32)
    model_fold0 = load_model(x_ph, 0)
    model_fold1 = load_model(x_ph, 1)

    # Load preprocessing
    preproc_fold0 = pickle.load(open(os.path.join(workdir, 'preproc_fold0.pickle'), 'rb'))
    preproc_fold1 = pickle.load(open(os.path.join(workdir, 'preproc_fold1.pickle'), 'rb'))

    # Create chain with friends
    d = make_dataset([filename], cfg.ntuples_base, cfg.friends_base, folder)
    num_entries = d.GetEntries()

    # Convert to numpy and stack to input dataset
    npy = ROOT.RDataFrame(d).AsNumpy(cfg.ml_variables + ['event'])
    inputs  = np.vstack([np.array(npy[k], dtype=np.float32) for k in cfg.ml_variables]).T

    # Apply model of fold 0 to data of fold 1 and v.v.
    mask_fold0 = npy['event'] % 2 == 0 #ACHTUNG
    mask_fold1 = npy['event'] % 2 == 1
    if np.sum(mask_fold0) + np.sum(mask_fold1) != num_entries:
        raise Exception('Events in folds dont add up to expected total')

    outputs_fold0 = session.run(model_fold0,
            feed_dict={x_ph: preproc_fold0.transform(inputs[mask_fold0])})
    scores_fold0 = outputs_fold0 #ACHTUNG
    #indices_fold0 = np.argmax(outputs_fold0, axis=1)

    outputs_fold1 = session.run(model_fold1,
            feed_dict={x_ph: preproc_fold1.transform(inputs[mask_fold1])})
    scores_fold1 = outputs_fold1 #ACHTUNG
    #indices_fold1 = np.argmax(outputs_fold1, axis=1)

    logger.info("Mask Fold 0 length: {}".format(len(mask_fold0)))
    logger.info("Mask Fold 1 length: {}".format(len(mask_fold1)))
    logger.info("Scores Fold 0 length: {}".format(len(scores_fold0)))
    logger.info("Scores Fold 0: {}".format(scores_fold0[:]))
    logger.info("Scores Fold 1 length: {}".format(len(scores_fold1)))

    # Merge scores back together
    scores = np.zeros(npy['event'].shape, dtype=np.float32)
    scores[mask_fold0] = scores_fold1 #ACHTUNG
    scores[mask_fold1] = scores_fold0

    #indices = np.zeros(npy['event'].shape, dtype=np.float32)
    #indices[mask_fold0] = indices_fold0 #ACHTUNG
    #indices[mask_fold1] = indices_fold1

    # Write output file
    path = os.path.join(workdir, 'MLScores', filename, folder + '.root')
    print('Write to file {}'.format(path))
    f = ROOT.TFile(path, 'RECREATE')
    if f == None:
        raise Exception('Failed to create file at location {}'.format(path))
    dir_ = f.mkdir(folder)
    dir_.cd()
    t = ROOT.TTree('ntuple', 'ntuple')
    val = array('f', [-999])
    #idx = array('f', [-999])
    bval = t.Branch('ml_score', val, 'ml_score/F')
    #bidx = t.Branch('ml_index', idx, 'ml_index/F')
    for i in range(scores.shape[0]):
        val[0] = scores[i]
        #idx[0] = indices[i]
        t.Fill()
    t.Write()
    f.Close()

    print('End application')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_collect_jobs.log'), logging.DEBUG)
    main(args)
