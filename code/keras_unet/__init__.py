# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:14:45 2021

@author: a0956
"""

name = "keras_unet"

__version__ = "0.1.2"
# TODO add __all__

# from packaging import version # requires 3rd party package
from distutils.version import LooseVersion # built-in package?

# if tensorflow 2.x is present use tf.keras instead of Keras
try:
    from tensorflow import __version__ as TF_VERSION
    # TF = version.parse(TF_VERSION) >= version.parse("2.0.0") # requires 3rd party package
    TF = LooseVersion(TF_VERSION) >= LooseVersion("2.0.0")
except ImportError:
    TF = False
print('-----------------------------------------')
if TF:
    print('keras-unet init: TF version is >= 2.0.0 - using `tf.keras` instead of `Keras`')
else:
    print('keras-unet init: TF version is < 2.0.0 or not present - using `Keras` instead of `tf.keras`')
print('-----------------------------------------')

from . import models
from . import losses
from . import metrics
from . import utils