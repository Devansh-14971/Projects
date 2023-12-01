##from random import shuffle
##
##
##
##class Cards:
##
##    global suites,values
##    suites = ['Diamonds','Spades','Clubs','Hearts']
##    values = ['A','1','2','3','4','5','6','7','8','9','10','J','Q','K']
##
##    def __init__(self):
##        pass
##
##class Deck(Cards):
##
##    def __init__(self):
##        Cards.__init__(self)
##        self.NewDeck = []
##
##        for i in suites:
##            for j in values:
##                self.NewDeck.append(j+' of '+i)
##
##    def PopCard(self):
##        if(len(self.NewDeck)==0):
##            print('No Cards in Deck')
##        else:
##            Popped = self.NewDeck.pop()
##            print('Card Removed is ',Popped)
##
##class ShuffleDeck(Deck):
##
##    def __init__(self):
##        Deck.__init__(self)
##
##    def Shuffle(self):
##        if(len(self.NewDeck)<52):
##            print('Cannot Shuffle Deck')
##        else:
##            shuffle(self.NewDeck)
##            return self.NewDeck
##
##    def PopCard(self):
##        if(len(self.NewDeck)==0):
##            return 'Cannot be popped'
##        else:
##            Pop = self.NewDeck.pop()
##            return Pop
##
##
##def RationalNum:
##    def __init__(self,n,d=1):
##        self.n = n
##        self.d = d
##    def __add__(self,other):
##        if !isinstance(other,RationalNum):
##            other = RationalNum(other)
##        n = self.n*other.d+self.d*other.n
##        d = self.d*other.d
##        return RationalNum(n,d)
##    def __str__(self):
##        return "{}/{}".format(self.n,self.d)
##
##
##
##
##
##
##objCards = Cards()
##objDeck = Deck()
##print(objCards)

import matplotlib.pyplot as plt
import numpy as np
x = np.array([1,2,3])
y = np.array([2,3,5])
plt.plot(x,y)
plt.show()



