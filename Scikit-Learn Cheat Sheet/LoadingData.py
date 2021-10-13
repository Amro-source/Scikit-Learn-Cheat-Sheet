# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:46:53 2021

@author: Zikantika
"""

import numpy as np

X = np.random.random((10,5))

y = np.array([ 1,2 ,4 ,5 ,6 ,7 ,8 ,8 ,8 ,8 ,8 ])

X[X < 0.7] = 0