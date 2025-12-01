import numpy as np
import matplotlib.pyplot as plt

data = [[1900, 2432],
        [1910, 2737],
        [1920, 3079],
        [1930, 3542],
        [1940, 3832],
        [1950, 4271],
        [1960, 4581],
        [1970, 4951],
        [1980, 5123],
        [1990, 5140],
        [2000, 5340],
        [2010, 5548],
        ]

x = np.array([row[0] for row in data], dtype=float)
y = np.array([row[1] for row in data], dtype=float)
xx = 2020

#--------------------------------------------------------------------------------

def get_newton_coefs(xs, ys):
    n = len(xs)
    dd = np.zeros((n, n))
    dd[:,0] = ys
    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (xs[i + j] - xs[i])
    coefs = dd[0,:].copy()
    return coefs

def newton_eval(xs, coefs, x_eval):
    n = len(coefs)
    result = 0.0
    mult = 1.0
    for k in range(n):
        result += coefs[k] * mult
        mult *= (x_eval - xs[k])
    return result

#--------------------------------------------------------------------------------

def get_moments(a, b, c, d):
    n = len(b)
    ac = a.copy().astype(float)
    bc = b.copy().astype(float)
    cc = c.copy().astype(float)
    dc = d.copy().astype(float)
    
    for i in range(1, n):
        w = ac[i] / bc[i-1]
        bc[i] = bc[i] - w * cc[i-1]
        dc[i] = dc[i] - w * dc[i-1]
    
    x = np.zeros(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in reversed(range(n-1)):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    return x

def get_spline_coefs(xs, ys):
    n = len(xs)
    h = np.diff(xs)
    
    a = np.zeros(n - 2)
    b = np.zeros(n - 2)
    c = np.zeros(n - 2)
    d = np.zeros(n - 2)
    for i in range(1, n - 1):
        idx = i - 1
        a[idx] = h[i - 1]
        b[idx] = 2 * (h[i - 1] + h[i])
        c[idx] = h[i]
        d[idx] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    
    M_inner = get_moments(np.concatenate(([0.0], a)), b, np.concatenate((c, [0.0])), d)
    M = np.zeros(n)
    M[1:-1] = M_inner
    return M

def evaluate_spline(xs, ys, M, x_eval):
    n = len(xs)
    if x_eval <= xs[0]:
        i = 0
    elif x_eval >= xs[-1]:
        i = n - 2
    else:
        i = np.searchsorted(xs, x_eval) - 1
    h = xs[i + 1] - xs[i]
    A = (xs[i + 1] - x_eval) / h
    B = (x_eval - xs[i]) / h
    S = A * ys[i] + B * ys[i + 1] + ((A ** 3 - A) * M[i] + (B ** 3 - B) * M[i + 1]) * (h ** 2) / 6.0
    return S

#--------------------------------------------------------------------------------

def build_design_matrix(xs, deg):
    rows = []
    for xval in xs:
        row = [1.0]
        for p in range(1, deg + 1):
            row.append(row[-1] * xval)
        rows.append(row)
    return np.array(rows, dtype=float)

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

def get_mnk_coefs(xs, ys, deg):
    A = build_design_matrix(xs, deg)
    AT = A.T
    ATA = AT @ A
    
    ys_vec = np.array(ys, dtype=float)
    ATy = AT @ ys_vec
    coefs = gauss_solve(ATA, ATy)
    return coefs

def polyval(coefs, x):
    val = 0.0
    p = 1.0
    for c in coefs:
        val += c * p
        p *= x
    return val

def get_error(deg):
    n = len(x)
    errs = []
    for i in range(n):
        xt = np.delete(x, i)
        yt = np.delete(y, i)

        c = get_mnk_coefs(xt, yt, deg)
        pred = polyval(c, x[i])
        errs.append((pred - y[i])**2)
    return np.sqrt(np.mean(errs))

#--------------------------------------------------------------------------------

coefs_newton = get_newton_coefs(x, y)
y_ans = newton_eval(x, coefs_newton, xx)

print(f"- Ньютон (полином степени {len(x) - 1}): {y_ans}")

M = get_spline_coefs(x, y)
y_ans = evaluate_spline(x, y, M, xx)

print(f"- Кубический сплайн: {y_ans}")

max_deg = 10
degrees = list(range(0, max_deg + 1))
errors = [get_error(d) for d in degrees]
best_deg = min(degrees, key=lambda d: (errors[d], d))

coefs_mnk = get_mnk_coefs(x, y, best_deg)
y_ans = polyval(coefs_mnk, xx)

print(f"- Метод наименьших квадратов (полином степени {best_deg}): {y_ans}")

xs_plot = np.linspace(1900, 2025, 625)
ys_newton_plot = np.array([newton_eval(x, coefs_newton, xi) for xi in xs_plot])
ys_spline_plot = np.array([evaluate_spline(x, y, M, xi) for xi in xs_plot])
ys_mnk_plot = np.array([polyval(coefs_mnk, xi) for xi in xs_plot])

plt.figure(figsize=(10,6))
plt.plot(x, y, 'o', label='Данные')
plt.plot(xs_plot, ys_newton_plot, label='Ньютон')
plt.plot(xs_plot, ys_spline_plot, label='Кубический сплайн')
plt.plot(xs_plot, ys_mnk_plot, label='МНК')
plt.title('Популяция Дании: данные и аппроксимации')
plt.xlabel('Год')
plt.ylabel('Население (тыс. чел.)')
plt.legend()
plt.grid(True)
plt.show()