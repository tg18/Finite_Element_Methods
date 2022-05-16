import numpy as np
import sympy as sv
from sympy import *
import matplotlib.pyplot as plt
import collections


nsd=2 # number of dimensions in the problem
nx=6 # number of nodes along x and y axis
nnp=nx*nx # total number of nodes
nel=(nx-1)**2 # number of elements
nen=4 # number of nodes in each element
C=1 #thermal conductivity

x=np.linspace(0,1,num=nx)
y=np.linspace(0,1,num=nx)
xv, yv = np.meshgrid(x, y)
xv=np.reshape(xv,(nnp))
yv=np.repeat(y, nx, axis=0)
XY=np.zeros((nnp,nsd))

XY[:,0]=xv
XY[:,1]=yv

## IEN array

b=[];c=[];d=[];e=[]

for i in range(nx-1):
    b.append(np.arange(nx*i+1,nx*i+nx))
    c.append(np.arange(nx * i + 2, nx * i + nx+1))
    d.append(np.arange(nx * i + 2+nx, nx * i + nx+1+nx))
    e.append(np.arange(nx * i + 1+nx, nx * i + nx+nx))

b=np.array(b)
b=np.reshape(b,(nx-1)**2)
c=np.array(c)
c=np.reshape(c,(nx-1)**2)
d=np.array(d)
d=np.reshape(d,(nx-1)**2)
e=np.array(e)
e=np.reshape(e,(nx-1)**2)

IEN=[b,c,d,e]
IEN=np.array(IEN,dtype=int)

## ID array

ID=np.ones((nx,nx),dtype=int)
ID[0,:]=0
ID[nx-1,:]=0
ID[:,0]=0
ID[:,nx-1]=0

ID[int(((nx/2)-1)):int((nx/2+1)),int(((nx/2)-1)):int((nx/2+1))]=0
ID=np.reshape(ID,nnp)

## boundary array g
neq=collections.Counter(ID)[1.0]
h=1/(nx-1)

f=1
for i in range(nnp):
    if ID[i]==1:
        ID[i]=f
        f=f+1

g=np.ones_like(ID,dtype=int)
for i in range(nnp):
    if ID[i]==0:
        g[i]=1
    else:
        g[i]=0
g=np.reshape(g,(nx,nx))
g[round(0.4/h):round((0.6/h))+1,round(0.4/h):round((0.6/h))+1]=0
g=np.reshape(g,nnp)

## LM array

LM=np.zeros((nen,(nx-1)**2),dtype=int)

for i in range(4):
    for j in range((nx-1)**2):
        LM[i,j]=ID[IEN[i,j]-1]

## Generating mesh

x1,y1=np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,nx))
plt.plot(x1, y1,color='k')
plt.plot(np.transpose(x1), np.transpose(y1),color='k')
plt.title('MESH')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

## SHAPE FUNCTION SUBROUTINE

z = sv.symbols("z")
n = sv.symbols("n")
n1 = (1 - z)*(1-n) / 4
n2 = (1 + z)*(1-n) / 4
n3 = (1 + z)*(1+n) / 4
n4 = (1 - z)*(1+n) / 4

def n_func(i):
    if i==0:
        return (1 - z)*(1-n) / 4
    elif i==1:
        return (1 + z)*(1-n) / 4
    elif i==2:
        return (1 + z)*(1+n) / 4
    elif i==3:
        return (1 - z)*(1+n) / 4

N1 = lambdify([z,n], n1, 'numpy')
N2 = lambdify([z,n], n2, 'numpy')
N3 = lambdify([z,n], n3, 'numpy')
N4 = lambdify([z,n], n4, 'numpy')
N=[N1,N2,N3,N4]
n1_z = n1.diff(z)
N1_z = lambdify([z,n], n1_z, 'numpy')
n1_n = n1.diff(n)
N1_n = lambdify([z,n], n1_n, 'numpy')
n2_z = n2.diff(z)
N2_z = lambdify([z,n], n2_z, 'numpy')
n2_n = n2.diff(n)
N2_n = lambdify([z,n], n2_n, 'numpy')
n3_z = n3.diff(z)
N3_z = lambdify([z,n], n3_z, 'numpy')
n3_n = n3.diff(n)
N3_n = lambdify([z,n], n3_n, 'numpy')
n4_z = n4.diff(z)
N4_z = lambdify([z,n], n4_z, 'numpy')
n4_n = n4.diff(n)
N4_n = lambdify([z,n], n4_n, 'numpy')

N_z=[N1_z,N2_z,N3_z,N4_z]
N_n=[N1_n,N2_n,N3_n,N4_n]

## Finding d array

k=np.zeros(((nel,nen,nen)))
K=np.zeros((neq,neq))

f=np.zeros(((nel,nen,1)))
F=np.zeros((neq,1))

for i in range(nel):

    X=((1 - z)*(1-n) / 4)*XY[IEN[0,i]-1,0]+((1 + z)*(1-n) / 4)*XY[IEN[1,i]-1,0]+((1 + z)*(1+n) / 4)*XY[IEN[2,i]-1,0]+((1 - z)*(1+n) / 4)*XY[IEN[3,i]-1,0]
    Y = ((1 - z)*(1-n) / 4)* XY[IEN[0, i]-1,1] + ((1 + z)*(1-n) / 4) * XY[IEN[1, i]-1,1] + ((1 + z)*(1+n) / 4) * XY[IEN[2, i]-1,1] + ((1 - z)*(1+n) / 4) * XY[IEN[3, i]-1,1]
    X1 = lambdify([z, n], X, 'numpy')
    Y1 = lambdify([z, n], Y, 'numpy')
    x=[X1,Y1]
    x1_z = X.diff(z)
    X1_z = lambdify([z, n], x1_z, 'numpy')
    x1_n = X.diff(n)
    X1_n = lambdify([z, n], x1_n, 'numpy')
    y1_z = Y.diff(z)
    Y1_z = lambdify([z, n], y1_z, 'numpy')
    y1_n = Y.diff(n)
    Y1_n = lambdify([z, n], y1_n, 'numpy')
    X_Z=[X1_z,X1_n]
    Y_Z=[Y1_z,Y1_n]
    j=x1_z*y1_n-x1_n*y1_z
    J=lambdify([z,n],j,'numpy')

    N_x=np.zeros((nel,nsd,nen))
    n1_x=[(n1_z*y1_n-n1_n*y1_z)/j,(n1_n*x1_z-n1_z*x1_n)/j]
    N1_x=lambdify([z,n],n1_x,'numpy')
    n2_x = [(n2_z * y1_n - n2_n * y1_z) / j, (n2_n * x1_z - n2_z * x1_n) / j]
    N2_x = lambdify([z, n], n2_x, 'numpy')
    n3_x = [(n3_z * y1_n - n3_n * y1_z) / j, (n3_n * x1_z - n3_z * x1_n) / j]
    N3_x = lambdify([z, n], n3_x, 'numpy')
    n4_x = [(n4_z * y1_n - n4_n * y1_z) / j, (n4_n * x1_z - n4_z * x1_n) / j]
    N4_x = lambdify([z, n], n4_x, 'numpy')
    N_x=[N1_x,N2_x,N3_x,N4_x]

    for j in range(nen):
        for l in range(nen):
            k[i,j,l]=C*integrate( np.matmul(N_x[j](z,n),N_x[l](z,n)), (z, -1, 1),(n,-1,1))

    for j in range(nen):
        f[i,j,0]=-(k[i,j,0]*g[IEN[0,i]-1]+k[i,j,1]*g[IEN[1,i]-1]+k[i,j,2]*g[IEN[2,i]-1]+k[i,j,3]*g[IEN[3,i]-1])

for i in range(nel):
    for j in range(nen):
        for l in range(nen):
            if LM[j,i]!=0 and LM[l,i]!=0:
                K[LM[j,i]-1,LM[l,i]-1]=K[LM[j,i]-1,LM[l,i]-1]+k[i,j,l]

        if LM[j,i]!=0:
            F[LM[j,i]-1,0]=F[LM[j,i]-1,0]+f[i,j,0]

D = np.linalg.solve(K, F)
D=np.array(D)

## Finding temperature u values on whole mesh

def d(e,index):
    if LM[index,e]==0:
        return 0
    else:
        return D[LM[index,e]-1]


u=np.array(g,dtype=float)

for i in range(nel):
    U=((1 - z)*(1-n) / 4)*d(i,0)+((1 + z)*(1-n) / 4)*d(i,1)+((1 + z)*(1+n) / 4)*d(i,2)+((1 - z)*(1+n) / 4)*d(i,3)
    if U==0:
        U = lambdify([z, n], U, 'numpy')
    else:
        U = lambdify([z, n], U[0], 'numpy')
    for j in range(nen):
        if LM[j, i] != 0:
            if j == 0:
                u[IEN[j, i] - 1] = U(-1, -1)
            elif j == 1:
                u[IEN[j, i] - 1] = U(1, -1)
            elif j == 2:
                u[IEN[j, i] - 1] = U(1, 1)
            elif j == 3:
                u[IEN[j, i] - 1] = U(-1, 1)

u=u.reshape(nx,nx)
print(u, u.shape)
feature_x = np.arange(0, nx)
feature_y = np.arange(0, nx)

# Creating 2-D grid of features
[X, Y] = np.meshgrid(feature_x, feature_y)
fig, ax = plt.subplots(1, 1)


# plots contour lines
CS=ax.contourf(X, Y, u)
cbar = fig.colorbar(CS)

ax.set_title('Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

