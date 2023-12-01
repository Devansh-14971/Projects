def Fibbo(n:int):
    if(n in [1,2]):
        return 1
    x = [1,[0,1]]
    for i in range(3,n+1):
        x[1][0] = x[1][1]
        x[1][1] = x[0]
        x[0] = x[1][0] + x[1][1]
    return x[0]
print(Fibbo(7))