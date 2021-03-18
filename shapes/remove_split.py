import os
import argparse

import ROOT
from utils import Reader
from utils.config import control_variables, analysis_variable, analysis_categories

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
    r = Reader([os.path.join(args.workdir, 'shapes_main.root')])
    f = ROOT.TFile(os.path.join(args.workdir, 'shapes_nosplit.root'), 'RECREATE')
    for variable in [v + '_inclusive' for v in control_variables] + \
                    [analysis_variable + '_' + c for c in analysis_categories]:
    
        for type in ['Nominal', 'same_sign']:
            try:
                ## Data ##
                data = r.get('data', type, variable)
                data.Write()

                ## TT ##
                data_tt = r.get('ttt', type, variable)
                for process in ['ttl', 'ttj']:

                    h = r.get(process, type, variable)
                    data_tt.Add(h, +1)

                name = str(data_tt.GetName()).replace('ttt', 'tt')
                data_tt.SetNameTitle(name, name)
                data_tt.Write()

                ## VV ##
                data_vv = r.get('vvt', type, variable)
                for process in ['vvl', 'vvj']:

                    h = r.get(process, type, variable)
                    data_vv.Add(h, +1)

                name = str(data_vv.GetName()).replace('vvt', 'vv')
                data_vv.SetNameTitle(name, name)
                data_vv.Write()

                ## ZL ##
                data_zl = r.get('zl', type, variable)
                for process in ['zj']:

                    h = r.get(process, type, variable)
                    data_zl.Add(h, +1)
                data_zl.Write()

                ## ZTT ##
                data_ztt = r.get('ztt', type, variable)
                data_ztt.Write()

                ## W ##
                data_w = r.get('w', type, variable)
                data_w.Write()

                ## ggh ##
                data_ggh = r.get('ggh', type, variable)
                data_ggh.Write()

                ## qqh ##
                data_qqh = r.get('qqh', type, variable)
                data_qqh.Write()
            except:
                pass


    for type in ['jpt_1_weightsUp', 'jpt_1_weightsDown']: 
        variable = 'ml_score_nll_cat'
        try:
            # ## Data ##
            # data = r.get('data', type, variable)
            # data.Write()

            ## TT ##
            data_tt = r.get('ttt', type, variable)
            for process in ['ttl', 'ttj']:

                h = r.get(process, type, variable)
                data_tt.Add(h, +1)

            name = str(data_tt.GetName()).replace('ttt', 'tt')
            data_tt.SetNameTitle(name, name)
            data_tt.Write()

            ## VV ##
            data_vv = r.get('vvt', type, variable)
            for process in ['vvl', 'vvj']:

                h = r.get(process, type, variable)
                data_vv.Add(h, +1)

            name = str(data_vv.GetName()).replace('vvt', 'vv')
            data_vv.SetNameTitle(name, name)
            data_vv.Write()

            ## ZL ##
            data_zl = r.get('zl', type, variable)
            for process in ['zj']:

                h = r.get(process, type, variable)
                data_zl.Add(h, +1)
            data_zl.Write()

            ## ZTT ##
            data_ztt = r.get('ztt', type, variable)
            data_ztt.Write()

            ## W ##
            data_w = r.get('w', type, variable)
            data_w.Write()

            ## ggh ##
            data_ggh = r.get('ggh', type, variable)
            data_ggh.Write()

            ## qqh ##
            data_qqh = r.get('qqh', type, variable)
            data_qqh.Write()
        except:
            pass
    f.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'qcd_estimation.log'), logging.INFO)
    main(args)