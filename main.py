import numpy as np
import random
import matplotlib.pyplot as plt

#регулярный план эксперимента y
def regularplan_y(y0,y1,n):
    ny = int(np.sqrt(n))
    y = np.linspace(y0, y1, ny)
    yv=np.meshgrid(y)
    return yv

def regularplan_x(x0,x1,n):
    nx = int(np.sqrt(n))
    x = np.linspace(x0, x1, nx)
    xv = np.meshgrid(x)
    return xv

#матрица частных производных в экспериментальных точках x
def jacobian(x):
    """ регрессионная модель - гиперплоскость """
    num_points = x.shape[0]
    order = x.shape[1] + 1
    jac = np.zeros((num_points, order))
    for row in range(num_points):
        jac[row, 0] = 1
        jac[row, 1:] = x[row,:]
    return jac

#случайный эксперимент
def randomplan(x0, x1, y0, y1, n):
    x = range(0,2*n)
    plan = []
    for i in range(n):
        plan.append([
            x0 + (x1-x0)*x[2*i],
            y0 + (y1-y0)*x[2*i+1]
        ])
    return np.array(plan)

#параметры 3D-гауссоиды и координаты точек
k1=regularplan_x(1,3,1000)
k2=regularplan_y(2,5,1000)
X, Y = np.meshgrid(k1, k2)
#параметры функции
def f(alfa,x, y):
    e=1.5
    k=alfa[0]
    return e*np.sin(2*k*x + 2*np.pi*y + 30)

#параметры трехмерной гуассоиды без шумов
fig = plt.figure()
ax = plt.axes(projection='3d')
Z = f([1,2,3,1,0.5],X, Y)
ax = plt.axes(projection='3d')
ax.set_title('График исходной функции при отсутствии шумов')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis')
plt.show()

#Двумерный график регулярного плана эксперимента
plt.scatter(X, Y, s=10, c="g",linewidths=2)
plt.title('Заголовок графика')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.show()

#добавление шума
noise=0.1
noisez1=noise*np.random.normal(0,len([1,3]))
noisez2=noise*np.random.normal(0,len([2,5]))
noiseZ1,noiseZ2=np.meshgrid(noisez1,noisez2)
Noise=noiseZ1+noiseZ2
#расчет функции с добавлением шума
Z1=f([1,2,3,1,0.5],X, Y)+Noise
ax = plt.axes(projection='3d')
ax.set_title('График исходной функции с нормальным шумом')
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,cmap='magma')
plt.show()

#расчет достоверности
planb = randomplan(1,3,2,5, 14)
num_points = planb.shape[0]
ret = np.zeros(num_points)
response=ret + 0.1 * random.uniform( -1, 1)
J = jacobian(planb)
coeff = np.matmul(np.matmul( np.linalg.pinv(np.matmul(np.transpose(J), J)),np.transpose(J)),response)
print('Коэффициенты уравнения регрессии',coeff)
model = np.matmul(J, coeff)
print('Значение функции в точках плана эксперимента:',model)
da = np.mean((model-response)**2)
print('Дисперсия адекватности:',da)
dv = Noise**2
print('Дисперсия воспроизводимости',dv)
Fval = da/dv
print('F-критерий:',Fval)
fib = J.shape[0] - 1
print('fib',fib)
fia = J.shape[0] - J.shape[1]
print('fia',fia)
vals = [2.11552234, 2.63465046, 4.02451844]
print('Квантили распределения Фишера qF',vals)