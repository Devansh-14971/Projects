import numpy as np
class rect:
    def __init__(self,l,b):
        self.length = int(l)
        self.breadth = int(b)
        self.area = l*b
    def show(self):
        return np.ones((self.length,self.breadth))

a,b = map(int,input("Length and breadth{space seperated}:").split())
c = rect(a,b)
print(c.show(),c.length,c.breadth,c.area,sep = "\n")
