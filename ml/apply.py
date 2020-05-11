import os
import argparse

import ROOT
import numpy as np
np.random.seed(1234)
from sklearn.metrics import confusion_matrix
from utils import config as cfg
import tensorflow as tf
tf.set_random_seed(1234)

from array import array

from collect_events import make_dataset

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
    model_fold0 = tf.keras.models.load_model(os.path.join(args.workdir, 'model_fold0.h5'))
    model_fold1 = tf.keras.models.load_model(os.path.join(args.workdir, 'model_fold1.h5'))

    for process in cfg.files:
        logger.info('Process files of process {}'.format(process))
        for filename in cfg.files[process]:
            # Create chain with friends
            d = make_dataset([filename], cfg.ntuples_base, cfg.friends_base)
            num_entries = d.GetEntries()
            logger.info('Process file {} with {} events'.format(filename, num_entries))

            # Convert to numpy and stack to input dataset
            npy = ROOT.RDataFrame(d).AsNumpy(cfg.ml_variables + ['event'])
            inputs  = np.vstack([np.array(npy[k], dtype=np.float32) for k in cfg.ml_variables]).T

            # Apply model of fold 0 to data of fold 1 and v.v.
            mask_fold0 = npy['event'] % 2 == 0
            mask_fold1 = npy['event'] % 2 == 1
            logger.debug('Events in fold 0 / fold 1 / total: {} / {} / {}'.format(np.sum(mask_fold0), np.sum(mask_fold1), num_entries))
            if np.sum(mask_fold0) + np.sum(mask_fold1) != num_entries:
                logger.fatal('Events in folds dont add up to expected total')
                raise Exception

            outputs_fold0 = model_fold1.predict(inputs[mask_fold0])
            scores_fold0 = np.max(outputs_fold0, axis=1)

            outputs_fold1 = model_fold0.predict(inputs[mask_fold1])
            scores_fold1 = np.max(outputs_fold1, axis=1)

            # Merge scores back together
            scores = np.zeros(npy['event'].shape, dtype=np.float32)
            scores[mask_fold0] = scores_fold0
            scores[mask_fold1] = scores_fold1

            # Make output folder
            os.mkdir(os.path.join(args.workdir, 'MLScores', filename))

            # Write output file
            f = ROOT.TFile(os.path.join(args.workdir, 'MLScores', filename, filename + '.root'), 'RECREATE')
            dir_ = f.mkdir('mt_nominal')
            dir_.cd()
            t = ROOT.TTree('ntuple', 'ntuple')
            val = array('f', [-999])
            b = t.Branch('ml_score', val, 'ml_score/F')
            for x in scores:
                val[0] = x
                b.Fill()
            t.Write()
            f.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_apply.log'), logging.INFO)
    main(args)
