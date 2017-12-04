# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:28:04 2017

@author: Sean
"""

import DataGen
import linearSVM

dataGenerator = DataGen.DataGenerator(n_sample=50)
X,Y = dataGenerator.generating(ang0 = 2., 
                               n_loop= 0.4, 
                               rad0 = 0.3, 
                               rand = 0.8)

clf = linearSVM.SVMclassifier( n_iteration= 1000)

clf.fit(X, Y, 50)
clf.plot()
