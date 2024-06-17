from setuptools import setup

import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(SCRIPT_DIR / "assera/python/setuputils"))
import setuputils as utils

setup(ext_modules=[utils.CMakeExtension("_lang_python")],
      cmdclass=dict(build_ext=utils.CMakeBuild),
      use_scm_version=utils.scm_version("assera/python/assera/_version.py"))
