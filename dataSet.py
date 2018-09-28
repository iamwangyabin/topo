import num2pic
import sampler
import topopt2D
import shutil
import os
import matplotlib.pyplot as plt
#####################################################
#   这个模块是用来产生在一个指定大小的区域随即几个支撑和力   #
#                                                   #
#                                                   #
#####################################################
exis=[]
for j in range(3):
    for i in range(100):
        i=i+100*j
        topopt2D.mkdir('dataset/' + str(i))
        c = sampler.random_config()
        load = num2pic.createLoadImg(list(c['LOAD_NODE_Y']), list(c['LOAD_VALU_Y']), shape_x=40, shape_y=40,
                                     path='./dataset/' + str(i) + '/load.png')
        support = num2pic.createSupportImg(list(c['FXTR_NODE_X']), list(c['FXTR_NODE_Y']), shape_x=40, shape_y=40,
                                           path='./dataset/' + str(i) + '/support.png')
        volfrac = 0.15
        rmin = 1.5  # lower number: more branching (initial: 5.4, try 2.0 or 1.5) proposal: 0.04 * nelx
        penal = 3.0  # ensure black and white solution
        ft = 1  # ft==0 -> sensitivity filtering, ft==1 -> density filtering
        chg = 0.1
        folder = 'dataset/' + str(i)  # mbb
        try:
            topopt2D.main(volfrac, penal, rmin, ft, chg, folder)
            exis.append(i)
            print(i)
        except BaseException:
            plt.close('all')
    for dir in os.listdir('dataset'):
        if int(dir) not in exis:
            shutil.rmtree('./dataset/'+dir)
