import numpy as np
class DataGenerator ():
    def __init__(self, n_sample=10):
        self.n_sample = n_sample

    def generating(self, ang0 = 0, n_loop= 2, rad0 = 0, rand = 0.4):
        detas = []
        for i in range(self.n_sample):
            angRange = np.pi * 2 *n_loop
            ang_i = ang0 + angRange*i/self.n_sample
            rad_i = rad0 + (1-rad0)*i/self.n_sample
            x0 = rad_i*np.cos(ang_i)
            y0 = rad_i*np.sin(ang_i)
            detas.append([x0, y0])

        detas = np.array(detas)
        detaRad = (1-rad0)/(angRange/2/np.pi)/2
        
        x = np.row_stack( (detas, detas*-1) )
        if rand:
            area = rand*detaRad
            x += (np.random.random(x.shape)-0.5)*area
        
        
        y = np.row_stack((np.zeros((self.n_sample,1)), np.ones((self.n_sample,1))))
        
        import random
        randInd = list(range(self.n_sample*2))
        random.shuffle(randInd)
        x, y = x[randInd,:], y[randInd,:]
        y = np.ravel(y)*2 - 1
        return x, y

#def dataPlot(data_x, data_y, predict_y = None):
#    from matplotlib import pyplot as plt
#    if predict_y is None:
#
#        for y in range(2):
#            ind = np.where(data_y==y)[0]
#            plt.scatter(data_x[ind, 0], data_x[ind, 1])
#        plt.show()
#    else:
#        plt.subplot(1,2,1)
#        for y in range(2):
#            ind = np.where(data_y==y)[0]
#            plt.scatter(data_x[ind, 0], data_x[ind, 1])
#        plt.subplot(1,2,2)
#        for y in range(2):
#            ind = np.where(predict_y==y)[0]
#            plt.scatter(data_x[ind, 0], data_x[ind, 1])
#        plt.show()
    
    
