### Ayush Bisen ###
### 21105025 ###

import numpy as np
import sympy as sv
from sympy import *
import matplotlib.pyplot as plt
import math

q=1
x_exact=np.linspace(0,1,num=4000)

fig=plt.figure()

## exact solution ##

u_exact=q/6*(1-x_exact**3)
u_deriv=-q*(x_exact**2)/2
ax1=plt.subplot()
ax1.plot(x_exact,u_exact,label='exact_solution')

## U_deriv function ##

def deriv(U_h,x):
  n_points=len(x)
  h=x[1]-x[0]
  derv=[]
  derv.append((U_h[1]-U_h[0])/h)
  for i in range(1,n_points-1):
    der=(U_h[i+1]-U_h[i-1])/(2*h)
    derv.append(der)
  derv.append((U_h[n_points-1]-U_h[n_points-2])/h)
  derv=np.array(derv)
  return derv

## U function for n=1 ##

def approximate_solution_for_n_1(n,nen):
    h = 1 / n;
    x1 = np.linspace(0, 1, n + 1)
    k = np.zeros((n, nen, nen))
    f = np.zeros((n, nen))

    z = sv.symbols("z")
    n1 = (1 - z) / 2
    n2 = (1 + z) / 2
    N1 = lambdify(z, n1, 'numpy')
    N1_z = n1.diff(z)
    N1_z = lambdify(z, N1_z, 'numpy')
    N2_z = n2.diff(z)

    for i in range(n):
        x = (h * z + x1[i] + x1[i + 1]) / 2
        X = lambdify(z, x, 'numpy')
        x_z = x.diff(z)
        X_z = lambdify(z, x_z, 'numpy')
        k[i, 0, 0] = integrate(N1_z(z) * N1_z(z) * 2, (z, -1, 1))
        f[i, 0] = integrate(N1(z) * q * X(z) * X_z(z), (z, -1, 1))

    K = np.zeros((n, n))
    F = np.zeros(n)

    for i in range(n):
        for j in range(n):
            K[i:i + nen, j:j + nen] = K[i:i + nen, j:j + nen] + k[i, :, :]
            F[i:i + nen] = F[i:i + nen] + f[i, :]
    F = F.reshape(n, 1)
    D = np.linalg.solve(K, F)
    U_1 = D[0] * (1 - x_exact)

    return U_1,x_exact

## U function for 'n' not equal to 1 ##

def approximate_solution(n,nen):
    q=1;h=1/n
    z = sv.symbols("z")
    x_exact = np.linspace(0, 1, num=4000)
    n1 = (1 - z) / 2
    n2 = (1 + z) / 2
    N1 = lambdify(z, n1, 'numpy')
    N2 = lambdify(z, n2, 'numpy')
    N1_z = n1.diff(z)
    N1_z = lambdify(z, N1_z, 'numpy')
    N2_z = n2.diff(z)
    N2_z = lambdify(z, N2_z, 'numpy')

    xnum = np.linspace(0, 1, n + 1)

    def N(x, num, xnum, h):
        if num == 0:
            if xnum[0] <= x <= xnum[1]:
                return (xnum[1] - x) / h
            else:
                return 0
        elif 1 <= num < n:
            if xnum[num - 1] <= x <= xnum[num]:
                return (x - xnum[num - 1]) / h
            elif xnum[num] < x <= xnum[num + 1]:
                return (xnum[num + 1] - x) / h
            else:
                return 0

    k = np.zeros((n, nen, nen))
    f = np.zeros((n, nen))
    for i in range(n):
        x = (h * z + xnum[i] + xnum[i + 1]) / 2
        X = lambdify(z, x, 'numpy')
        x_z = x.diff(z)
        X_z = lambdify(z, x_z, 'numpy')
        k[i, 0, 0] = integrate(N1_z(z) * N1_z(z) * (2 / h), (z, -1, 1))
        k[i, 0, 1] = integrate(N1_z(z) * N2_z(z) * (2 / h), (z, -1, 1))
        k[i, 1, 0] = integrate(N2_z(z) * N1_z(z) * (2 / h), (z, -1, 1))
        k[i, 1, 1] = integrate(N2_z(z) * N2_z(z) * (2 / h), (z, -1, 1))
        f[i, 0] = integrate(N1(z) * q * X(z) * X_z(z), (z, -1, 1))
        f[i, 1] = integrate(N2(z) * q * X(z) * X_z(z), (z, -1, 1))

    K = np.zeros((n, n))
    F = np.zeros(n)

    for i in range(n):
        if i == (n - 1):
            K[i, i] = K[i, i] + k[i, 0, 0]
            F[i] = F[i] + f[i, 0]
        else:
            K[i, i] = K[i, i] + k[i, 0, 0]
            K[i, i + 1] = K[i, i + 1] + k[i, 0, 1]
            K[i + 1, i] = K[i + 1, i] + k[i, 1, 0]
            K[i + 1, i + 1] = K[i + 1, i + 1] + k[i, 1, 1]
            F[i] = F[i] + f[i, 0]
            F[i + 1] = F[i + 1] + f[i, 1]

    F = F.reshape(n, 1)
    D = np.linalg.solve(K, F)

    U_h = np.zeros_like(x_exact)
    for j in range(n):
        for i in range(len(x_exact)):
            U_h[i] = U_h[i] + D[j] * (N(x_exact[i], j, xnum, h))

    ax2 = plt.subplot()
    return U_h

## plotting U_approximate and U_exact ##

for i in range(1,5):
    if i==1:
        U_h,x_exact=approximate_solution_for_n_1(i,1)
        ax1.plot(x_exact,U_h,label='N='+str(i))

    else:
        U_h=approximate_solution(i,2)
        ax1.plot(x_exact,U_h,label='N='+str(i))
plt.legend()
plt.title('Question B.1')
plt.xlabel('x')
plt.ylabel('U_h')
plt.show()

## Plotting U_approx_deriv vs U_exact_deriv ##

plt.figure()
ax2=plt.subplot()
ax2.plot(x_exact,u_deriv,label='exact_solution')
for i in range(1,5):
    if i == 1:
        U_h, x_exact = approximate_solution_for_n_1(i, 1)
        Uh_deriv = deriv(U_h, x_exact)
        ax2.plot(x_exact,Uh_deriv,label='N=' + str(i))

    else:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        ax2.plot(x_exact, Uh_deriv, label='N=' + str(i))
plt.legend()
plt.title('Question B.2')
plt.xlabel('x')
plt.ylabel('U_derivative')
plt.show()

plt.figure()
ax3=plt.subplot()

## calculating error for n=4 at midpoints and plotting re,x for n=4 ##

for i in [4]:
    if i == 1:
        U_h, x_exact = approximate_solution_for_n_1(i, 1)
        Uh_deriv = deriv(U_h, x_exact)
        error=(Uh_deriv-u_deriv)/(q/2)
        print('For n=',i,'error value at midpoint is ',abs(error[50]))
        ax3.plot(x_exact,error,label='N=' + str(i))

    else:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        error = (Uh_deriv - u_deriv)/(q/2)
        print('For n=', i, 'error value at midpoint of element 1 is ', abs(error[12]+error[13])/2)
        print('For n=', i, 'error value at midpoint of element 2 is ', abs(error[37]+error[38])/2)
        print('For n=', i, 'error value at midpoint of element 3 is ', abs(error[62]+error[63])/2)
        print('For n=', i, 'error value at midpoint of element 4 is ', abs(error[87]+error[88])/2)
        ax3.plot(x_exact, error, label='N=' + str(i))
plt.legend()
plt.title('Question A.4 for n=4')
plt.xlabel('x')
plt.ylabel('re,x')
plt.show()

## Plotting ln(re,x) v/s ln(h) for n=1,2,3,4 ##

plt.figure()
ax4=plt.subplot()
rex=np.zeros(4)
H=[abs(math.log(1)),abs(math.log(0.5)),abs(math.log(0.333)),abs(math.log(0.25))]
for i in range (1,5):
    if i == 1:
        U_h, x_exact = approximate_solution_for_n_1(i, 1)
        Uh_deriv = deriv(U_h, x_exact)
        error=(Uh_deriv-u_deriv)/(q/2)
        rex[0]=abs(math.log(abs(error[50])))

    elif i==2:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        error = (Uh_deriv - u_deriv)/(q/2)
        rex[1]=abs(math.log(abs(error[25])))

    elif i==3:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        error = (Uh_deriv - u_deriv)/(q/2)
        rex[2]=abs(math.log(abs(error[16])*(1/3)+abs(error[17])*(2/3)))

    elif i==4:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        error = (Uh_deriv - u_deriv)/(q/2)
        rex[3]=abs(math.log((abs(error[12])+abs(error[13]))/2))


ax4.plot(H,rex)
plt.title('Question A.5')
plt.xlabel('ln(h)')
plt.ylabel('ln(re,x)')
plt.show()

plt.figure()
ax5=plt.subplot()

## calculating error for n=10,50,100 at midpoints and plotting re,x vs x for n=10,50,100 ##

for i in [10,50,100]:
    if i == 1:
        U_h, x_exact = approximate_solution_for_n_1(i, 1)
        Uh_deriv = deriv(U_h, x_exact)
        error=(Uh_deriv-u_deriv)/(q/2)
        print('For n=',i,'error value at midpoint is ',abs(error[50]))
        ax5.plot(x_exact,error,label='N=' + str(i))

    else:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        error = (Uh_deriv - u_deriv)/(q/2)
        print('For n=', i, 'error value at midpoint of element  is ', abs(error[50]))
        ax5.plot(x_exact, error, label='N=' + str(i))
plt.legend()
plt.title('Question B.3 for n=10,50,100')
plt.xlabel('x')
plt.ylabel('re,x')
plt.show()

## Plotting ln(re,x) v/s ln(h) for n=10,50,100 ##

plt.figure()
ax6=plt.subplot()
rex=np.zeros(3)
H=[abs(math.log(1/10)),abs(math.log(1/50)),abs(math.log(1/100))]
for i in [10,50,100]:
    if i == 0:
        U_h, x_exact = approximate_solution(i, 2)
        Uh_deriv = deriv(U_h, x_exact)
        error=(Uh_deriv-u_deriv)/(q/2)
        rex[0]=abs(math.log(abs(error[5])))

    elif i==1:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        error = (Uh_deriv - u_deriv)/(q/2)
        rex[1]=abs(math.log(abs(error[1])))

    elif i==2:
        U_h = approximate_solution(i, 2)
        Uh_deriv=deriv(U_h,x_exact)
        error = (Uh_deriv - u_deriv)/(q/2)
        rex[2]=abs(math.log(abs(error[0])+abs(error[1])))


ax6.plot(H,rex)
plt.title('ln(re,x) vs ln(h) for n=10,50,100')
plt.xlabel('ln(h)')
plt.ylabel('ln(re,x)')
plt.show()

plt.figure()
ax5=plt.subplot()