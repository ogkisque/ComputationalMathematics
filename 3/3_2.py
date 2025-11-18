#!/usr/bin/env python3
"""
Решение двух систем нелинейных уравнений четырьмя методами (адаптация для систем):
  A) cos(x-1) + y = 0.5
     x - cos(y) = 3

  B) (x - 1.4)^2 - (y - 0.6)^2 = 1
     4.2x^2 + 8.8y^2 = 1.42
"""
import math
import numpy as np

def norm_inf(v):
    return np.linalg.norm(v)

def gauss_solve(A_in, b_in):
    A = A_in.astype(float).copy()
    b = b_in.astype(float).copy()
    n = A.shape[0]

    for k in range(n - 1):
        pivot = np.argmax(np.abs(A[k:n, k])) + k

        if pivot != k:
            A[[k, pivot], :] = A[[pivot, k], :]
            b[[k, pivot]] = b[[pivot, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def bracket_search_scalar(phi, left, right, nsteps=1000):
    xs = np.linspace(left, right, nsteps+1)
    prev = phi(xs[0])
    brackets = []
    for i in range(1, len(xs)):
        x = xs[i]
        cur = phi(x)

        if cur is None:
            prev = cur
            continue
        if prev is None:
            prev = cur
            continue
        if prev == 0.0:
            brackets.append((xs[i-1], xs[i-1]))
        elif prev*cur < 0:
            brackets.append((xs[i-1], xs[i]))
        prev = cur
    return brackets

def bisection_scalar(phi, a, b, tol=1e-12, maxiter=200):
    fa = phi(a)
    fb = phi(b)
    if fa == 0.0: return a, 0
    if fb == 0.0: return b, 0

    it = 0
    while (b - a) / 2.0 > tol and it < maxiter:
        c = 0.5 * (a + b)
        fc = phi(c)
        it += 1
        if fc == 0.0:
            return c, it
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return 0.5 * (a + b), it

def F1(vec):
    x, y = vec
    return np.array([math.cos(x-1) + y - 0.5,
                     x - math.cos(y) - 3.0], dtype=float)

def J1(vec):
    x, y = vec
    return np.array([
        [-math.sin(x-1), 1.0],
        [1.0, math.sin(y)]
    ], dtype=float)

def phi1_scalar(x):
    y = 0.5 - math.cos(x-1)
    return x - math.cos(y) - 3.0

def G1(vec):
    x, y = vec
    return np.array([3.0 + math.cos(y),
                     0.5 - math.cos(x - 1.0)], dtype=float)

def F2(vec):
    x, y = vec
    return np.array([(x - 1.4) ** 2 - (y - 0.6) ** 2 - 1,
                     4.2*(x ** 2) + 8.8*(y ** 2) - 1.42], dtype=float)

def J2(vec):
    x, y = vec
    return np.array([[2*x - 2.8, -2*y + 1.2],
                     [8.4*x, 17.6*y]], dtype=float)

def phi2_scalar(x):
    y = math.sqrt(((1.42 - 4.2*(x*x)) / 8.8))
    return (x - 1.4) ** 2 - (y - 0.6) ** 2 - 1

def G2(vec):
    x, y = vec
    return np.array([math.sqrt(1 + (y-0.6)*(y-0.6)) + 1.4,
                     math.sqrt((1.42 - 4.2*(x*x)) / 8.8)], dtype=float)

def newton_system(F, J, x0, tol=1e-12, maxiter=100):
    x = np.array(x0, dtype=float)
    for i in range(1, maxiter+1):
        Fx = F(x)
        Jx = J(x)
        #delta = np.linalg.solve(Jx, -Fx)
        delta = gauss_solve(Jx, -Fx)

        x_new = x + delta
        if norm_inf(delta) < tol:
            return x_new, i, True
        x = x_new
    return x, maxiter, False

def modified_newton_system(F, J, x0, tol=1e-12, maxiter=300):
    x = np.array(x0, dtype=float)
    J0 = J(x0)
    try:
        for i in range(1, maxiter+1):
            Fx = F(x)
            #delta = np.linalg.solve(J0, -Fx)
            delta = gauss_solve(J0, -Fx)

            x_new = x + delta
            if norm_inf(delta) < tol:
                return x_new, i, True
            x = x_new
    except Exception:
        return x, 0, False
    return x, maxiter, False

def solve_system1():
    print("\n=== Система 1 ===")
    
    brackets = bracket_search_scalar(phi1_scalar, -10.0, 10.0, nsteps=20)
    print("Найденные бракеты:", brackets[:10])

    # Половинное деление
    a, b = brackets[0]
    root_x, it = bisection_scalar(phi1_scalar, a, b, tol=1e-12, maxiter=200)
    root_y = 0.5 - math.cos(root_x - 1.0)
    print(f"Половинное деление: x={root_x}, y={root_y}  (iter {it}) ||F||={norm_inf(F1([root_x,root_y]))}")

    # Простая итерация
    print("Простая итерация:", end=' ')
    x0 = np.array([3.0, 0.0])
    x = x0.copy()
    seq = [x.copy()]
    for i in range(1,301):
        x = G1(x)
        seq.append(x.copy())
        if norm_inf(seq[-1] - seq[-2]) < 1e-10:
            print(f"Сошёлся за {i} итераций: {x}  ||F||={norm_inf(F1(x))}")
            break
    else:
        print("Не сошёлся за 300 итераций. Последние приближения и остатки:")
        for k in range(len(seq)-5, len(seq)):
            print(f"    n={k}: {seq[k]}, ||F||={norm_inf(F1(seq[k]))}")

    # Метод Ньютона
    x0_newton = np.array([3.0, 1.0])
    root_newt, it_newt, conv_newt = newton_system(F1, J1, x0_newton)
    print(f"Метод Ньютона от x0={x0_newton}: root={root_newt}, iter={it_newt}, ||F||={norm_inf(F1(root_newt))}")

    # Модифицированный метод Ньютона
    root_mod, it_mod, conv_mod = modified_newton_system(F1, J1, x0_newton)
    print(f"Модифицированный метод Ньютона от x0={x0_newton}: root={root_mod}, iter={it_mod}, ||F||={norm_inf(F1(root_mod))}")

def solve_system2():
    print("\n=== Система 2 ===")

    brackets = bracket_search_scalar(phi2_scalar, -0.5, 0.5, nsteps=10)
    print("Найденные бракеты:", brackets[:10])

    # Половинное деление
    a, b = brackets[0]
    root_x, it = bisection_scalar(phi2_scalar, a, b, tol=1e-12, maxiter=200)
    root_y = math.sqrt(((1.42 - 4.2*(root_x*root_x)) / 8.8))
    print(f"Половинное деление: x={root_x}, y={root_y}  (iter {it}) ||F||={norm_inf(F2([root_x,root_y]))}")

    # Простая итерация
    # print("Простая итерация:", end=' ')
    # x0 = np.array([0.35, 0.35])
    # x = x0.copy()
    # seq = [x.copy()]
    # for i in range(1,301):
    #     x = G2(x)
    #     seq.append(x.copy())
    #     if norm_inf(seq[-1] - seq[-2]) < 1e-10:
    #         print(f"Сошёлся за {i} итераций: {x} ||F||={norm_inf(F2(x))}")
    #         break
    # else:
    #     print("Не сошёлся за 300 итераций. Последние приближения и остатки:")
    #     for k in range(len(seq)-5, len(seq)):
    #         print(f"n={k}: {seq[k]}, ||F||={norm_inf(F2(seq[k]))}")

    # Метод Ньютона
    x0_newton = np.array([0.5, 0.5])
    root_newt, it_newt, conv_newt = newton_system(F2, J2, x0_newton)
    print(f"Метод Ньютона от x0={x0_newton}: root={root_newt}, iter={it_newt}, ||F||={norm_inf(F2(root_newt))}")

    # Модифицированный Метод Ньютона
    root_mod, it_mod, conv_mod = modified_newton_system(F2, J2, x0_newton)
    print(f"Модифицированный Метод Ньютона от x0={x0_newton}: root={root_mod}, iter={it_mod}, ||F||={norm_inf(F2(root_mod))}")

if __name__ == "__main__":
    solve_system1()
    solve_system2()