import os
import argparse

import Dumbledraw.dumbledraw as dd
import Dumbledraw.styles as styles

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


def main(args, variable):
    # Config
    linear = True
    bkg_processes = ['qcd', 'vvt', 'vvl', 'vvj', 'w', 'ttt', 'ttl', 'ttj', 'zj', 'zl', 'ztt']
    category = name.split('_')[-1]
    variable = name.replace('_' + category, '')

    # Read histograms
    hists = {}
    reader = Reader([os.path.join(args.workdir, f) for f in ['shapes_main.root', 'shapes_qcd.root']])
    total_bkg = None
    for process in bkg_processes:
        h = reader.get(process, 'Nominal', name)
        hists[process] = h
        if not total_bkg:
            total_bkg = h.Clone()
        else:
            total_bkg.Add(h)

    hists['data'] = reader.get('data', 'Nominal', name)
    hists['ggh'] = reader.get('ggh', 'Nominal', name)
    hists['qqh'] = reader.get('qqh', 'Nominal', name)

    # Blinding
    is_blinded = False
    if args.blinding and category != 'inclusive':
        hists['data'] = hists[bkg_processes[0]].Clone()
        for process in bkg_processes[1:] + ['ggh', 'qqh']:
            hists['data'].Add(hists[process])
        is_blinded = True

    # Set backgrounds
    plot = dd.Plot([0.3 if linear else 0.5, [0.3, 0.28]], 'ModTDR', r=0.04, l=0.14, width=600)

    for process in bkg_processes:
        plot.add_hist(hists[process], process, 'bkg')
        plot.setGraphStyle(process, 'hist', fillcolor=styles.color_dict[process.upper()])

    plot.add_hist(total_bkg, 'total_bkg')
    plot.setGraphStyle('total_bkg', 'e2', markersize=0, fillcolor=styles.color_dict['unc'], linecolor=0)

    # Set data
    plot.add_hist(hists['data'], 'data_obs')
    data_norm = plot.subplot(0).get_hist('data_obs').Integral()
    plot.subplot(0).get_hist('data_obs').GetXaxis().SetMaxDigits(4)
    for i in [0, 2]:
        plot.subplot(i).setGraphStyle('data_obs', 'e0')

    # Set signals
    for i in [0,2] if linear else [1,2]:
        ggH = hists['ggh'].Clone()
        qqH = hists['qqh'].Clone()
        if ggH.Integral() > 0:
            ggH_scale = 0.5*data_norm/ggH.Integral()
        else:
            ggH_scale = 0.0
        if qqH.Integral() > 0:
            qqH_scale = 0.5*data_norm/qqH.Integral()
        else:
            qqH_scale = 0.0
        if i in [0,1]:
            ggH.Scale(ggH_scale)
            qqH.Scale(qqH_scale)
        plot.subplot(i).add_hist(ggH, 'ggH')
        plot.subplot(i).add_hist(ggH, 'ggH_top')
        plot.subplot(i).add_hist(qqH, 'qqH')
        plot.subplot(i).add_hist(qqH, 'qqH_top')

    plot.subplot(0 if linear else 1).setGraphStyle('ggH', 'hist',
            linecolor=styles.color_dict['ggH'], linewidth=3)
    plot.subplot(0 if linear else 1).setGraphStyle('ggH_top', 'hist', linecolor=0)
    plot.subplot(0 if linear else 1).setGraphStyle('qqH', 'hist',
            linecolor=styles.color_dict['qqH'], linewidth=3)
    plot.subplot(0 if linear else 1).setGraphStyle('qqH_top', 'hist', linecolor=0)

    # Make ratio
    bkg_ggH = plot.subplot(2).get_hist('ggH')
    bkg_qqH = plot.subplot(2).get_hist('qqH')
    bkg_ggH.Add(plot.subplot(2).get_hist('total_bkg'))
    bkg_qqH.Add(plot.subplot(2).get_hist('total_bkg'))
    plot.subplot(2).add_hist(bkg_ggH, 'bkg_ggH')
    plot.subplot(2).add_hist(bkg_ggH, 'bkg_ggH_top')
    plot.subplot(2).add_hist(bkg_qqH, 'bkg_qqH')
    plot.subplot(2).add_hist(bkg_qqH, 'bkg_qqH_top')
    plot.subplot(2).setGraphStyle('bkg_ggH', 'hist',
            linecolor=styles.color_dict['ggH'], linewidth=3)
    plot.subplot(2).setGraphStyle('bkg_ggH_top', 'hist', linecolor=0)
    plot.subplot(2).setGraphStyle('bkg_qqH', 'hist',
            linecolor=styles.color_dict['qqH'], linewidth=3)
    plot.subplot(2).setGraphStyle('bkg_qqH_top', 'hist', linecolor=0)
    plot.subplot(2).normalize(
            ['total_bkg', 'bkg_ggH', 'bkg_ggH_top', 'bkg_qqH', 'bkg_qqH_top', 'data_obs'],
            'total_bkg')

    # Stack background processes
    plot.create_stack(bkg_processes, 'stack')

    # Set axes limits and labels
    plot.subplot(0).setYlims(0, 1.8 * plot.subplot(0).get_hist('data_obs').GetMaximum())
    plot.subplot(2).setYlims(0.75, 1.35)

    if not linear:
        plot.subplot(1).setYlims(0.1, 0)
        plot.subplot(1).setYlabel('')
    if variable in styles.x_label_dict['mt']:
        x_label = styles.x_label_dict['mt'][variable]
    else:
        x_label = variable
    plot.subplot(2).setXlabel(x_label)
    plot.subplot(0).setYlabel('N_{events}')

    plot.subplot(2).setYlabel('')

    plot.scaleYLabelSize(0.8)
    plot.scaleYTitleOffset(1.1)

    # Wraw subplots
    # Argument contains names of objects to be drawn in corresponding order
    plot.subplot(0).Draw(
            ['stack', 'total_bkg', 'ggH', 'ggH_top', 'qqH', 'qqH_top', 'data_obs'])
    if not linear:
        plot.subplot(1).Draw(['stack', 'total_bkg', 'data_obs'])
    plot.subplot(2).Draw(
            ['total_bkg', 'bkg_ggH', 'bkg_ggH_top', 'bkg_qqH', 'bkg_qqH_top', 'data_obs'])

    # Create legends
    suffix = ['', '_top']
    for i in range(2):
        plot.add_legend(width=0.6, height=0.15)
        for process in reversed(bkg_processes):
            plot.legend(i).add_entry(0, process,
                    styles.legend_label_dict[process.upper().replace('TTL', 'TT').replace('VVL', 'VV')], 'f')
        plot.legend(i).add_entry(0, 'total_bkg', 'Bkg. stat. unc.', 'f')
        plot.legend(i).add_entry(0 if linear else 1, 'ggH%s' % suffix[i], '%s #times gg#rightarrowH'%str(int(ggH_scale)), 'l')
        plot.legend(i).add_entry(0 if linear else 1, 'qqH%s' % suffix[i], '%s #times qq#rightarrowH'%str(int(qqH_scale)), 'l')
        plot.legend(i).add_entry(0, 'data_obs', 'Data', 'PE')
        plot.legend(i).setNColumns(3)
    plot.legend(0).Draw()
    plot.legend(1).setAlpha(0.0)
    plot.legend(1).Draw()

    for i in range(2):
        plot.add_legend(reference_subplot=2, pos=1, width=0.5, height=0.03)
        plot.legend(i + 2).add_entry(0, 'data_obs', 'Data', 'PE')
        plot.legend(i + 2).add_entry(0 if linear else 1, 'ggH%s' % suffix[i], 'ggH+bkg.', 'l')
        plot.legend(i + 2).add_entry(0 if linear else 1, 'qqH%s' % suffix[i], 'qqH+bkg.', 'l')
        plot.legend(i + 2).add_entry(0, 'total_bkg', 'Bkg. stat. unc.', 'f')
        plot.legend(i + 2).setNColumns(4)
    plot.legend(2).Draw()
    plot.legend(3).setAlpha(0.0)
    plot.legend(3).Draw()

    # Draw additional labels
    plot.DrawLumi('59.7 fb^{-1} (2018, 13 TeV)')
    tag = category + ' (blinded)' if is_blinded else category
    plot.DrawChannelCategoryLabel('%s, %s' % ('#mu#tau_{h}', tag), begin_left=None)

    plot.save(os.path.join(args.workdir, '%s_%s.%s' % (variable, category, 'png')))
    plot.save(os.path.join(args.workdir, '%s_%s.%s' % (variable, category, 'pdf')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('--blinding', default=True, help='Apply blinding for all categories except inclusive')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'plot_control.log'), logging.INFO)
    for name in [v + '_inclusive' for v in control_variables] + \
                    [analysis_variable + '_' + c for c in analysis_categories]:
        main(args, name)
