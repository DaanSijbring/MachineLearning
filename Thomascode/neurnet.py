import numpy as np
class neuralnet:
    def __init__(self,fvectors,doutput,steps):
        self.x = [np.append(fvector,[1]) for fvector in fvectors]
        self.r = doutput
        self.w = self.onelayer(self.x,self.r,steps)
        self.y = np.dot(self.x,self.w)
        self.y = [y if y <= 10 else 10 for y in self.y]
        self.y = [y if y >= 1 else 1 for y in self.y]
    
    def genupdate(self,w,r,y,x):
        P = len(w)
        dw = (1/P)*(r - y)*x
        return(w + dw)

    def performance(self):
        samplevar = np.var(self.r)
        netvar = np.mean((self.y - self.r)**2)
        Expl = 1 - netvar/samplevar
        print(f'Variance model:{netvar},variance  sample:{samplevar},perc explained {Expl}')


    def onelayer(self,x,r,steps):
        w = np.zeros(len(x[1]))
        print(len(x[1]));print(len(w))
        for _ in range(steps):
            for j in range(len(x)):
                y = np.dot(w,x[j])
                w = self.genupdate(w,r[j],y,x[j])
        return(w)

