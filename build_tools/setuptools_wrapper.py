import os
from setuptools import build_meta as st_build_meta
from setuptools.build_meta import *


def build_wheel(wheel_dir, config_settings=None, meta_dir=None):
    import numpy as np
    os.environ["CFLAGS"] = "-I{0:s}".format(np.get_include())
    return st_build_meta.build_wheel(wheel_dir, config_settings, meta_dir)


def build_sdist(sdist_dir, config_settings=None):
    import numpy as np
    os.environ["CFLAGS"] = "-I{0:s}".format(np.get_include())
    return st_build_meta.build_sdist(sdist_dir, config_settings)


def build_editable(editable_dir, config_settings=None, meta_dir=None):
    import numpy as np
    os.environ["CFLAGS"] = "-I{0:s}".format(np.get_include())
    return st_build_meta.build_editable(editable_dir, config_settings,
                                        meta_dir)
