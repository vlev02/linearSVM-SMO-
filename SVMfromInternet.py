# -*- coding: utf-8 -*
from numpy import *
import pylab as pl
class my_svm(object):
    def __init__(self,filename,c=6,tol=0.001,miter=30):
        self.filename = filename;
        self.C = c;
        self.tol = tol;
        self.miter = miter;
        self.support_vector = [];
    def loadDataSet(self):
        dataMat = []; labelMat = [];
        fr = open(self.filename);
        for line in fr.readlines():
            lineArr = line.strip().split('\t');
            dataMat.append([float(lineArr[0]), float(lineArr[1])]);
            labelMat.append(float(lineArr[2]));
        return mat(dataMat),mat(labelMat).transpose();
    def rand_select_j(self,i):
        j=i;
        while j==i:
            j = int(random.uniform (self.m));
        return j;
    def sample_svm(self):
        '''alphs*Y*<X,Xi>+b'''
        self.X,self.Y = self.loadDataSet();
        self.m,self.n = shape(self.X);
        self.alpha = mat(zeros((self.m,1)));
        self.b=iter=0;
        while iter<self.miter:
            alpha_change=0;
            for i in range(self.m):
                '''求解Xi的预测值和误差'''
                '''multiply和矩阵乘法是不一样的'''
                Xi = float(multiply(self.alpha,self.Y).T*(self.X*self.X[i,:].T))+self.b;
                err_i = Xi - float(self.Y[i]);
                if (err_i*self.Y[i]<-self.tol and self.alpha[i]<self.C) or (err_i*self.Y[i]>self.tol and self.alpha[i]>0):
                    j = self.rand_select_j(i);
                    '''随机选择另一个确定其误差，SMO的关键就是选择两个变量同时变化'''
                    Xj = float(multiply(self.alpha,self.Y).T*(self.X*self.X[j,:].T))+self.b;
                    err_j = Xj - float(self.Y[j]);
                    alpha_i_old,alpha_j_old = self.alpha[i].copy(), self.alpha[j].copy();
                    '''求解H和L'''
                    if self.Y[i] == self.Y[j]:
                        L = float(max(0,self.alpha[i]+self.alpha[j]-self.C));
                        H = float(min(self.C,self.alpha[i]+self.alpha[j]));
                    else:
                        L = float(max(0,self.alpha[j]-self.alpha[i]));
                        H = float(min(self.C,self.C+self.alpha[j]-self.alpha[i]));
                    if L == H:
                        continue;
                    '''alpha的增量为：Y2*(err_1-err_2)/(K11+K22-2K12)统计学习方法上有详细的证明，其中K是核函数'''
                    eta = float(self.X[i,:]*self.X[i,:].T+self.X[j,:]*self.X[j,:].T-2.0*self.X[i,:]*self.X[j,:].T);
                    if 0==eta:
                        continue;
                    self.alpha[j] += self.Y[j]*(err_i-err_j)/eta;
                    '''根据限制条件：0<=alpha_j<=C,确定最终的alpha_j'''
                    if self.alpha[j] > H:
                        self.alpha[j] = H;
                    if self.alpha[j] < L:
                        self.alpha[j] = L;
                    #print("alpha[j]: ",float(alpha[j]),"alpha_j_old: ",float(alpha_j_old));
                    if (abs(float(self.alpha[j])-float(alpha_j_old))<0.00001):
                        '''alpha的变化太小'''
                        #print("alpha的变化太小");
                        continue;
                    '''两个alpha变化大小相同，单方向相反'''
                    self.alpha[i] += self.Y[j]*self.Y[i]*(alpha_j_old-self.alpha[j]);
                    '''下面确定b，主要是通过新的alpha_i和alpha_j来确定b,主要运用两个公式，统计学习方法（130）'''
                    b1 = self.b - err_i- self.Y[i]*(self.alpha[i]-alpha_i_old)*self.X[i,:]*self.X[i,:].T - self.Y[j]*(self.alpha[j]-alpha_j_old)*self.X[i,:]*self.X[j,:].T
                    b2 = self.b - err_j- self.Y[i]*(self.alpha[i]-alpha_i_old)*self.X[i,:]*self.X[j,:].T - self.Y[j]*(self.alpha[j]-alpha_j_old)*self.X[j,:]*self.X[j,:].T
                    if (0 < self.alpha[i]) and (self.C > self.alpha[i]):
                        b = b1
                    elif (0 < self.alpha[j]) and (self.C > self.alpha[j]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2)/2.0
                    alpha_change = alpha_change + 1;
            if 0 == alpha_change:
                iter+=1;
            else:
                iter = 0;
        self.__calculate_support_vector_and_weight_();
    def __calculate_support_vector_and_weight_(self):
        '''我们根据KKT条件给出支持向量，也就是alpha不等于0的项'''
        '''我们根据公式为：alpha*Y*X求解w'''
        self.w = zeros((self.n,1));
        for i in range(self.m):
            if self.alpha[i]:
                self.support_vector.append([self.X[i].getA()[0][0],self.X[i].getA()[0][1]]);
                self.w += multiply(self.alpha[i]*self.Y[i],self.X[i,:].T);
        self.support_vector = mat(self.support_vector);
    def plot_svm(self):
        X = []; Y = [];
        fr = open(self.filename);
        for line in fr.readlines():
            lineArr = line.strip().split('\t');
            X.append([float(lineArr[0]), float(lineArr[1])]);
            Y.append(float(lineArr[2]));
        X=mat(X);
        a = -self.w[0]/self.w[1];
        XX = linspace(-5, 15);
        YY = a * XX - (self.b[0].getA1()[0])/self.w[1];
        pl.plot(XX, YY, 'k-');
        pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired);
        pl.axis('tight');
        pl.show();