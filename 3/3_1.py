import math

def f1(x):
    return x*x - 20*math.sin(x)

def df1(x):
    return 2*x - 20*math.cos(x)

def f2(x):
    return x * (2 ** x) - 1

def df2(x):
    return 2 ** x + x * (2 ** x) * math.log(2)

def bisection(f, a, b, tol=1e-12, maxiter=200):
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a, 0
    if fb == 0.0:
        return b, 0

    it = 0
    while (b - a) / 2.0 > tol and it < maxiter:
        c = 0.5 * (a + b)
        fc = f(c)
        it += 1
        if fc == 0.0:
            return c, it
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return 0.5 * (a + b), it

def g1(x):
    return math.sqrt(20 * math.sin(x))

def g2(x):
    return 1 / (2 ** x)

def simple_iter(g, x0, tol=1e-12, maxiter=50):
    x = x0
    for n in range(1, maxiter + 1):
        if (math.sin(x) < 0):
            return 55555, 55555
        x_new = g(x)
        x = x_new
        if (x < tol):
            return x, n
    return x, maxiter

def newton(f, df, x0, tol=1e-12, maxiter=200):
    x = x0
    for i in range(1, maxiter+1):
        d = df(x)
        if d == 0:
            return x, i
        x_new = x - f(x) / d
        if abs(x_new - x) < tol:
            return x_new, i
        x = x_new
    return x, maxiter

def modified_newton(f, df, x0, tol=1e-12, maxiter=200):
    d0 = df(x0)
    x = x0
    for i in range(1, maxiter+1):
        x_new = x - f(x) / d0
        if abs(x_new - x) < tol:
            return x_new, i
        x = x_new
    return x, maxiter

def find_sign_changes(f, left=0.0, right=20.0, npoints=1000):
    xs = [left + (right - left) * i / (npoints - 1) for i in range(npoints)]
    vals = [f(x) for x in xs]
    brackets = []
    for i in range(len(xs) - 1):
        if vals[i] == 0.0:
            brackets.append((xs[i], xs[i]))
        elif vals[i] * vals[i+1] < 0:
            brackets.append((xs[i], xs[i+1]))
    return brackets

print("1 уравнение")
print("Поиск отрезков с корнями на отрезке [0, 20]...")
brackets = find_sign_changes(f1, 0.0, 20.0, npoints=20)
for i, br in enumerate(brackets[:10]):
    print(f"  {i+1}: {br}")

nonzero_br = None
for br in brackets:
    if not (abs(br[0]) < 1e-15 and abs(br[1]) < 1e-15):
        nonzero_br = br
        break

a, b = nonzero_br

# 1) Половинное деление
root_bisect, it_b = bisection(f1, a, b, tol=1e-12, maxiter=500)
print(f"\nПоловинное деление: x = {root_bisect} (итераций: {it_b})")
print(f"  f(x) = {f1(root_bisect)}")

# 2) Простая итерация
root_simple_iter, it_simple_iter = simple_iter(g1, x0=(a + b) / 2)
print(f"\nПростая итерация: x = {root_simple_iter} (итераций: {it_simple_iter})")
print(f"  f(x) = {f1(root_simple_iter)}")

# 3) Метод Ньютона
root_newton, it_newt = newton(f1, df1, x0=(a+b)/2, tol=1e-12, maxiter=500)
print(f"\nМетод Ньютона: x = {root_newton}, итераций = {it_newt}")
print(f"  f(x) = {f1(root_newton)}")

# 4) Модифицированный метод Ньютона
root_mod, it_mod = modified_newton(f1, df1, x0=(a+b)/2, tol=1e-12, maxiter=500)
print(f"\nМодифицированный метод Ньютона: x = {root_mod}, итераций = {it_mod}")
print(f"  f(x) = {f1(root_mod)}")

print("\n2 уравнение")
print("Поиск отрезков с корнями на отрезке [0, 20]...")
brackets = find_sign_changes(f2, 0.0, 20.0, npoints=20)
for i, br in enumerate(brackets[:10]):
    print(f"  {i+1}: {br}")

nonzero_br = None
for br in brackets:
    if not (abs(br[0]) < 1e-15 and abs(br[1]) < 1e-15):
        nonzero_br = br
        break

a, b = nonzero_br

# 1) Половинное деление
root_bisect, it_b = bisection(f2, a, b, tol=1e-12, maxiter=500)
print(f"\nПоловинное деление: x = {root_bisect} (итераций: {it_b})")
print(f"  f(x) = {f2(root_bisect)}")

# 2) Простая итерация
root_simple_iter, it_simple_iter = simple_iter(g2,x0=(a+b)/2)
print(f"\nПростая итерация: x = {root_simple_iter} (итераций: {it_simple_iter})")
print(f"  f(x) = {f2(root_simple_iter)}")

# 3) Метод Ньютона
root_newton, it_newt = newton(f2, df2, x0=(a+b)/2, tol=1e-12, maxiter=500)
print(f"\nМетод Ньютона: x = {root_newton}, итераций = {it_newt}")
print(f"  f(x) = {f2(root_newton)}")

# 4) Модифицированный метод Ньютона
root_mod, it_mod = modified_newton(f2, df2, x0=(a+b)/2, tol=1e-12, maxiter=500)
print(f"\nМодифицированный метод Ньютона: x = {root_mod}, итераций = {it_mod}")
print(f"  f(x) = {f2(root_mod)}")