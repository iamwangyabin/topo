from __future__ import print_function

import numpy as np
import topopt2D
import random


for i in range(20):
    volfrac=random.uniform(0.1,0.5)
    penal=random.uniform(2,4)
    rmin=random.uniform(1.5,3)
    ft = 1
    chg = 0.1
    folder = 'testData/'
    savf='testData/'
    topopt2D.main(volfrac, penal, rmin, ft, chg, folder,savf)




if __name__ == "__main__":

    topopt2D.main(volfrac, penal, rmin, ft, chg, folder)