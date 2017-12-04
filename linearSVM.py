# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:44:18 2017

@author: Sean
"""
import numpy as np

class SVMclassifier():
    def __init__(self, KernelType = None, n_iteration= 1000):
        self.n_ite = n_iteration
        if KernelType == None:
            self.kernelFun = lambda x1, x2 :np.dot(x1, x2)
        return
    
    def Kij(self, i, j):
        return self.kernelFun(self.Data_X[i], self.Data_X[j])
    
    def fit(self, data_X, data_Y, C,):
        self.epsilon = 0.001
        self.C = C
        self.Data_X = np.array(data_X)
        self.Data_Y = np.array(data_Y)
        self.n_sample, self.n_feature = self.Data_X.shape
        self.w, self.b = np.zeros(self.n_feature), 0
        self.SMO()
        
        
        return
    
    def SMO(self,):
        self.tempAlpha = np.zeros(self.n_sample)
        count = 0
        while count < self.n_ite:
            print(count)
            predicts = np.dot(self.w, self.Data_X.transpose()) + self.b
            self.mu = predicts
            self.E = predicts-self.Data_Y
            flag = self.selectIJ()
            if flag is False:
                print ('!!!!!!!!!!!!!!!!!!!!!!!!count=',count)
                return 
            else:
                count += 1
        return

    def updataWB(self, i, j):
        w = np.dot(self.Data_Y*self.tempAlpha, self.Data_X)
        bi = self.Data_Y[i] - w.dot(self.Data_X[i])
        bj = self.Data_Y[j] - w.dot(self.Data_X[j])
        if self.tempAlpha[i]>0 and self.tempAlpha[i]<self.C:
            return w,bi
        if self.tempAlpha[j]>0 and self.tempAlpha[j]<self.C:
            return w,bj
        return w,(bi+bj)/2
 
    def selectIJ(self):
        alphaIList = []
        for i in range(self.n_sample):
            if (self.E[i]*self.Data_Y[i]<-self.epsilon and self.tempAlpha[i]<self.C) \
            or (self.E[i]*self.Data_Y[i]>self.epsilon and self.tempAlpha[i]>0):
                alphaIList.append(i)
        if len(alphaIList) is 0:
            alphaIList = range(self.n_sample)
            for i in alphaIList:
                if self.selectJ_(i,[]): return True
        else:
            for i in alphaIList:
                if (self.tempAlpha[i]>0) and (self.tempAlpha[i]<self.C):
                    self.selectJ_(i,alphaIList)
                    if self.selectJ_(i,[]): return True
            for i in alphaIList:
                self.selectJ_(i,alphaIList)
                if self.selectJ_(i,[]): return True
        return False

    def selectJ_(self, i, IList):
        # 1
        if self.E[i]<0:
            j = np.argmax(self.E)
        else:
            j = np.argmin(self.E)
        if self.updateAlpha(i,j): return True
        # 2
        for j in IList:
            if self.updateAlpha(i,j): return True
        # 3
        for j in range(self.n_sample):
            if self.updateAlpha(i,j): return True
        return False
    
    def updateAlpha(self, i,j):
        if i == j:
            return False
        self.s = self.Data_Y[i]*self.Data_Y[j]
        alphaJ_LH = self.alphaLH(i,j)
        if alphaJ_LH[0] == alphaJ_LH[1]:
            return False
        
        eta = self.Kij(i,i) + self.Kij(j,j) - self.Kij(i,j)*2
        if eta == 0:
            return False
        if eta > 0:
            alphaJ = self.tempAlpha[j] + self.Data_Y[j]*(self.E[i]-self.E[j])/eta
            if alphaJ > alphaJ_LH[1]:
                alphaJ = alphaJ_LH[1]
            elif alphaJ < alphaJ_LH[0]:
                alphaJ = alphaJ_LH[0]
            alphaI = self.tempAlpha[i] - self.s*(alphaJ-self.tempAlpha[j])
        if eta < 0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!eta<0')
            alphaI_LH = self.tempAlpha[i] - self.s*(alphaJ_LH-self.tempAlpha[j])
            yv1 = self.Data_Y[i]*(self.mu[i]-self.b)- \
                 self.tempAlpha[i]*self.Kij(i,i) - \
                 self.s*self.tempAlpha[j]*self.Kij(i,j)
            yv2 = self.Data_Y[j]*(self.mu[j]-self.b)- \
                 self.tempAlpha[j]*self.Kij(j,j) - \
                 self.s*self.tempAlpha[i]*self.Kij(i,j)
            PhiComp = 0.5*self.Kij(i,i)*alphaI_LH*alphaI_LH+ \
                      0.5*self.Kij(j,j)*alphaJ_LH*alphaJ_LH+ \
                      self.s*self.Kij(i,j)*alphaI_LH*alphaJ_LH+ \
                      yv1*alphaI_LH+yv2*alphaJ_LH
            ind = np.argmin(PhiComp)
            alphaI, alphaJ = alphaI_LH[ind], alphaJ_LH[ind]
        
        if self.tempAlpha[j] == alphaJ:
            return False
        self.tempAlpha[i], self.tempAlpha[j] = alphaI, alphaJ
        self.w, self.b = self.updataWB(i, j)
        print('ij->',i,j,self.Data_Y[i], self.Data_Y[j],self.tempAlpha[i],self.tempAlpha[j])
        return True
    
    def alphaLH(self, i, j):
        iLimit = [self.s*self.tempAlpha[i]+self.tempAlpha[j]]
        iLimit.append( iLimit[0] - self.s*self.C )
        iLimit = sorted(iLimit)
        L = max(0, iLimit[0])
        H = min(self.C, iLimit[1])
        return (L, H)
    
    def plot(self):
        if self.n_feature ==2:
            self.Plot_2D()
        if self.n_feature ==3:
            self.Plot_3D()
    def Plot_3D(self):
        import matplotlib.pyplot as plt  
        from mpl_toolkits.mplot3d import Axes3D  
        
        fig = plt.figure()
        ax = Axes3D(fig) 
        
        ind = np.where(self.Data_Y == 1)[0]
        coord = self.Data_X[ind]
        ax.scatter(coord[:,0], coord[:,1], coord[:,2], c = 'r', marker = '^' )
        
        ind = np.where(self.Data_Y == -1)[0]
        coord = self.Data_X[ind]
        ax.scatter(coord[:,0], coord[:,1], coord[:,2], c = 'b', marker = 'o' )
        
        coordPlaneMin = [np.min(self.Data_X[:,i]) for i in range(3)]
        coordPlaneMax = [np.max(self.Data_X[:,i]) for i in range(3)]
        coordPlaneM = np.array([coordPlaneMin,coordPlaneMax])
        print (coordPlaneM)
        X = np.linspace(coordPlaneM[0,0], coordPlaneM[1,0], 10)
        Y = np.linspace(coordPlaneM[0,1], coordPlaneM[1,1], 10)
        X, Y = np.meshgrid(X, Y)
        Z = -(self.w[0]*X+self.w[1]*Y+self.b)/self.w[2]
        for i in range(10):
            ax.plot3D(X[i,:],Y[i,:],Z[i,:], 'gray') 
            ax.plot3D(X[:,i],Y[:,i],Z[:,i], 'gray') 
        plt.show()
    def Plot_2D(self):
        import matplotlib.pyplot as plt  

        ind = np.where(self.Data_Y == 1)[0]
        coord = self.Data_X[ind]
        plt.scatter(coord[:,0], coord[:,1], c = 'r', marker = '^' )
        
        ind = np.where(self.Data_Y == -1)[0]
        coord = self.Data_X[ind]
        plt.scatter(coord[:,0], coord[:,1], c = 'b', marker = 'o' )
        
        coordPlaneMin = [np.min(self.Data_X[:,i]) for i in range(2)]
        coordPlaneMax = [np.max(self.Data_X[:,i]) for i in range(2)]
        coordPlaneM = np.array([coordPlaneMin,coordPlaneMax])
        X = None
        print('w = ',self.w)
        if self.w[1] != 0:
            X = np.array([coordPlaneM[0,0], coordPlaneM[1,0]])
            Y = -(self.w[0]*X + self.b)/self.w[1]
        elif self.w[0] != 0:
            Y = np.array([coordPlaneM[0,1], coordPlaneM[1,1]] )
            X = -(self.w[1]*Y + self.b)/self.w[0]
        
        # margin
        if X is not None:
            plt.plot(X, Y, 'k-')
            deta = self.w/(self.w.dot(self.w))
            plt.plot(X+deta[0], Y+deta[1], 'r--')
            plt.plot(X-deta[0], Y-deta[1], 'b--')
            plt.xlim( coordPlaneM[0,0]-0.1, coordPlaneM[1,0]+0.1 )    # set the xlim to xmin, xmax
            plt.ylim( coordPlaneM[0,1]-0.1, coordPlaneM[1,1]+0.1 ) 
        plt.show()

        
        
        