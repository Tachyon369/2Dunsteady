import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import math


@jit(nopython=True)  # cuda acceleration comment out if using only cpu, works with amd graphics but buggy.
def timeiter(T, dr, dt, alpha, n):
    ap_t = dr**2 / (alpha * dt)
    T1 = np.zeros(n)
    T1[n-1] = T[n-1]
    for i in range(1, n-1):
        rp = dr * i
        ap = ap_t - (2 * rp / dr)
        aw = ((2 * rp) - dr) / (2 * dr)
        ae = ((2 * rp) + dr) / (2 * dr)
        T1[i] = ((ap * T[i]) + (aw * T[i-1]) + (ae * T[i+1])) / ap_t
    T1[0] = T1[1]   # Flux condition
    return T1


def timeloop(T, dr, dt, alpha, n, tim, out_bound, graphtype):
    ite = 0
    Tn = np.zeros(n)
    x = np.zeros(n)
    for i in range(0, n):
        x[i] = i * dr
    if graphtype == "circular":
        plt_graph(T, n, out_bound, ite, dt)
    else:
        plt_graphline(T, ite, x, n, dt)
    for i in range(0, n):
        Tn[i] = T[i]
    while(ite * dt) < tim:
        ite = ite + 1
        Tn = timeiter(Tn, dr, dt, alpha, n)
        if graphtype == "circular":
            plt_graph(Tn, n, out_bound, ite, dt)
        else:
            plt_graphline(Tn, ite, x, n, dt)
        print(str(round(100*ite * dt / tim, 3)), "% complete")


@jit(nopython=True)  # cuda acceleration comment out if using only cpu, works with amd graphics but buggy.
def calc_graph(Tn, n, out_bound):
    u = np.zeros((2 * n + 1, 2 * n + 1))
    for i in range(0, 2 * n + 1):
        for j in range(0, 2 * n + 1):
            Rad_plt = math.sqrt((n - i) ** 2 + (n - j) ** 2)
            if int(Rad_plt) >= n:
                u[i][j] = out_bound
            if int(Rad_plt) < n:
                u[i][j] = Tn[int(Rad_plt)]
    return u


def plt_graph(Tn, n, out_bound, ite, dt):
    u = calc_graph(Tn, n, out_bound)
    pic_id = "plots/" + str(ite) + ".png"
    plt.title('Temperature Contour Plot of time:' + str(format(ite*dt, '.5f')) + 'sec')
    plt.xlabel('x from 0 to 2')
    plt.ylabel('y from 0 to 2')
    plt.imshow(u, cmap='jet')
    plt.colorbar()
    plt.savefig(pic_id, dpi=300)
    plt.close()


def plt_graphline(Tn, ite, x, n, dt):
    pic_id = "plots/" + str(ite) + ".png"
    plt.plot(x, Tn)
    plt.xlabel('radial distance from 0 to 1')
    plt.ylabel('Temperature (K)')
    plt.title('T vs R curve of time:' + str(format(ite*dt, '.5f')) + 'sec')
    plt.savefig(pic_id, dpi=300)
    plt.close()

def main():
    r = 1
    alpha = 0.00001
    dr = 0.01
    dt = 0.0001
    tim = 0.1
    ini_temp = 1000.0
    out_bound = 300.0
    graphtype = "line"  # can use "circular" or "line"
    if dt > (dr ** 3 / (2 * r * alpha)):
        print("reduce dt or increase dr!")
        exit()
    n = int((r / dr) + 1)
    T = np.zeros(n)
    # setting initial boundary conditions
    for i in range(0, n-1):
        T[i] = ini_temp
    T[n-1] = out_bound
    timeloop(T, dr, dt, alpha, n, tim, out_bound, graphtype)


main()










