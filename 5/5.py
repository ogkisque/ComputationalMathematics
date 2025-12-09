import numpy as np
from math import sqrt

x_nodes = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
y_nodes = np.array([-0.333,  0.0,   -0.125, -0.056, 0.0,   0.046,  0.083,  0.115,  0.143])

h = x_nodes[1] - x_nodes[0]

def g(xx):
    res = 0
    j = np.searchsorted(x_nodes, xx) - 1
    x0, x1 = x_nodes[j], x_nodes[j + 1]
    y0, y1 = y_nodes[j], y_nodes[j + 1]
    t = (xx - x0) / (x1 - x0)
    res = y0 * (1 - t) + y1 * t
    return res

def rectangle_method_mid():
    vals = []
    for i in range(len(y_nodes) - 1):
        vals.append((y_nodes[i] + y_nodes[i+1]) / 2)
    return h * np.sum(vals)

def rectangle_method_left():
    vals = []
    for i in range(len(y_nodes) - 1):
        vals.append(y_nodes[i])
    return h * np.sum(vals)

def rectangle_method_right():
    vals = []
    for i in range(len(y_nodes) - 1):
        vals.append(y_nodes[i+1])
    return h * np.sum(vals)

def trapezoid_method():
    ys = np.asarray(y_nodes)
    return h * (0.5 * ys[0] + np.sum(ys[1:-1]) + 0.5 * ys[-1])

def simpson_method():
    ys = np.asarray(y_nodes)
    return h / 3 * (ys[0] + ys[-1] + 4 * np.sum(ys[1::2]) + 2 * np.sum(ys[2:-1:2]))

GAUSS_NODES_WEIGHTS = {
    2: (np.array([-1.0/sqrt(3), 1.0/sqrt(3)]), np.array([1.0, 1.0])),
    3: (np.array([-sqrt(3/5), 0.0, sqrt(3/5)]), np.array([5/9, 8/9, 5/9])),
    4: (np.array([-0.861136, -0.339981, 0.339981, 0.861136]),
        np.array([0.347855, 0.652145, 0.652145, 0.347855]))
}

def gauss_helper(a, b, m):
    nodes, weights = GAUSS_NODES_WEIGHTS[m]
    t = 0.5 * (b + a) + 0.5 * (b - a) * nodes
    vals = g(t)
    return 0.5 * (b - a) * np.sum(weights * vals)

def gauss_method(m):
    total = 0.0
    for i in range(len(x_nodes) - 1):
        a, b = x_nodes[i], x_nodes[i + 1]
        total += gauss_helper(a, b, m)
    return total

if __name__ == '__main__':
    rect_mid = rectangle_method_mid()
    rect_right = rectangle_method_right()
    rect_left = rectangle_method_left()
    trap = trapezoid_method()
    simpson = simpson_method()

    gauss_comp = {m: gauss_method(m) for m in (2,3,4)}

    print(f"Шаг h = {h}")
    print(f"1.1) Прямоугольники (по серединам): {rect_mid:.4f}")
    print(f"1.2) Прямоугольники (по правым): {rect_right:.4f}")
    print(f"1.3) Прямоугольники (по левым): {rect_left:.4f}")
    print(f"2) Трапеции: {trap:.4f}")
    print(f"3) Симпсон: {simpson:.4f}")
    print("4) Гауссова квадратура:")
    for m, val in gauss_comp.items():
        print(f"    {m} точки: {val:.4f}")
