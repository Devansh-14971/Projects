#solve for N queens
n = int(input())
class board(self):
    def __init__(self,n):
        board = [[0 for i in range(n)] for x in range(n)]
    def update(self,(i,c)):
    	j,k = 0,0
        while i<n:
            self[i][k]=1
            i+=1
        while c<n:
            self[j][c]=1
            c+=1
        #down right
        j,k = i,c
        while(i<n and c<n):
            self[i][c]=1
            i+=1
            c+=1
        #up right
        j,k = i,c
        while(i<n and c<n):
            self[i][c]=1
            i+=1
            c-=1
        #down left
        j,k = i,c
        while(i<n and c<n):
            self[i][c]=1
            i-=1
            c+=1
        #up left
        j,k = i,c
        while(i<n and c<n):
            self[i][c]=1
            i-=1
            c-=1
            
        
def Nqueens(i,b,n):
    for c in range(n):
        if b[i][c]==0:
            b[i][c] = q
board = board()
if __name__ == '__main__':
    Nqueens(0,board,n)
