# from a given list and target find pair that sums upto taarget
class pairs:
    def find_pair(l,tgt):
        l2 = {}
        l3 = []
        for i,n in enumerate(l):
            t = tgt-n
            if t in l2.keys():
                l3.append([n,t])
                l2.remove(l2[t])
            l2[n] = i
        return l3
a = list(map(int,input("List(space seperated):").split(' ')))
target = int(input("Target:"))
print(pairs.find_pair(a,target))
