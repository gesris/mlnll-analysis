import ROOT

import logging
logger = logging.getLogger("")


class Reader:
    def __init__(self, filenames):
        self._filenames = filenames
        self._objs = {}
        self._files = []
        for filename in self._filenames:
            f = ROOT.TFile(filename)
            if f == None:
                logger.fatal("Failed to open file {}".format(filename))
                raise Exception
            self._read_all(f)
            self._files.append(f)

    def _read_all(self, file_):
        for key in file_.GetListOfKeys():
            name = key.GetName()
            h = file_.Get(name)
            if h == None:
                logger.fatal("Failed to get {}".format(name))
                raise Exception
            if name in self._objs:
                logger.fatal("Object {} already loaded".format(name))
                raise Exception
            self._objs[name] = h

    def get(self, process, variation, variable):
        keys = []
        for key in self._objs:
            dataset = key.split('#')[0]
            selections = key.split('#')[1]
            selections_ = selections.split('-')
            variation_ = key.split('#')[2]
            variable_ = key.split('#')[3]
            if variable_ == variable and variation_ == variation and (process in selections or dataset == process):
                keys.append(key)
        if len(keys) == 0:
            logger.fatal("Failed to find object for process {} with variation {} and variable {}".format(
                process, variation, variable))
            raise Exception
        elif len(keys) != 1:
            logger.fatal("Found more than one object for process {} with variation {} and variable {}".format(
                process, variation, variable))
        return self._objs[keys[0]]
