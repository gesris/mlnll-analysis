import ROOT

import logging
logger = logging.getLogger("")


class Reader:
    def __init__(self, filename):
        self._filename = filename
        self._file = ROOT.TFile(filename)
        if not self._file:
            logger.fatal("Failed to open file {}".format(self._filename))
            raise Exception
        self._objs = self._get_all()

    def _get_all(self):
        objs = {}
        for key in self._file.GetListOfKeys():
            name = key.GetName()
            h = self._file.Get(name)
            if not h:
                logger.fatal("Failed to get {} from file {}".format(self._filename, name))
                raise Exception
            objs[name] = h
        return objs

    def __getitem__(self, name):
        h = self._file.Get(name)
        if not h:
            logger.fatal("Failed to get {} from file {}".format(self._filename, name))
            raise Exception
        return h

    def get(self, process, variation, variable):
        for key in self._objs:
            dataset = key.split('#')[0]
            selections = key.split('#')[1]
            process_ = selections.split('-')[-1]
            variation_ = key.split('#')[2]
            variable_ = key.split('#')[3]
            if variable_ == variable and variation_ == variation and (process_ == process or dataset == process):
                return self._objs[key]
        logger.fatal("Failed to find object for process {} with variation {} and variable {}".format(
            process, variation, variable))
        raise Exception
