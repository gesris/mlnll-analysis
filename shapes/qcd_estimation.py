import ROOT
from utils import Reader

import logging
logger = logging.getLogger("")


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main():
    r = Reader('shapes.root')
    variable = 'm_vis'

    data = r.get('data', 'same_sign', variable)
    for process in ['zl', 'zj', 'w', 'ttt', 'ttj', 'ttl', 'vvt', 'vvj', 'vvl']:
        h = r.get(process, 'same_sign', variable)
        data.Add(h, -1)

    name = str(data.GetName()).replace('data', 'qcd').replace('same_sign', 'Nominal')
    data.SetNameTitle(name, name)

    f = ROOT.TFile('shapes_qcd.root', 'RECREATE')
    data.Write()
    f.Close()


if __name__ == "__main__":
    setup_logging('qcd_estimation.log', logging.INFO)
    main()
