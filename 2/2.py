import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)

def build_A(n):
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
            else:
                A[i, j] = 1.0 / ( (i+1) + (j+1) )
    return A


def build_b(n):
    return np.array([1.0 / (i+1) for i in range(n)], dtype=float)

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

def lu_decompose(A):
    A = np.array(A, dtype=float)
    n, m = A.shape

    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)

    for k in range(n):
        for j in range(k, n):
            s = 0.0
            for p in range(k):
                s += L[k, p] * U[p, j]
            U[k, j] = A[k, j] - s

        for i in range(k + 1, n):
            s = 0.0
            for p in range(k):
                s += L[i, p] * U[p, k]
            L[i, k] = (A[i, k] - s) / U[k, k]

    return L, U

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def lu_solve(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)

    L, U = lu_decompose(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x

def jacobi(A, b, x0=None, tol=1e-10, maxiter=5000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    D = np.diag(A)

    residuals = []
    for k in range(maxiter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            x_new[i] = (b[i] - s) / D[i]

        r = np.linalg.norm(A @ x_new - b)
        residuals.append(r)

        if r < tol:
            return x_new, residuals, k + 1

        x = x_new

    return x, residuals, maxiter

def gauss_seidel(A, b, x0=None, tol=1e-10, maxiter=5000):
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.astype(float).copy()
    residuals = []

    for k in range(maxiter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        r = np.linalg.norm(A @ x - b)
        residuals.append(r)
        if r < tol:
            return x, residuals, k+1
    return x, residuals, maxiter

def sor(A, b, omega=1.25, x0=None, tol=1e-10, maxiter=5000):
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.astype(float).copy()
    residuals = []
    for k in range(maxiter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x_i_new = (b[i] - s1 - s2) / A[i, i]
            x[i] = (1 - omega) * x_old[i] + omega * x_i_new
        r = np.linalg.norm(A @ x - b)
        residuals.append(r)
        if r < tol:
            return x, residuals, k+1
    return x, residuals, maxiter

def gradient_descent(A, b, x0=None, tol=1e-10, maxiter=5000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    residuals = []
    for k in range(maxiter):
        Ax = A @ x
        r = b - Ax
        res_norm = np.linalg.norm(Ax - b)
        residuals.append(res_norm)
        if res_norm < tol:
            return x, residuals, k

        Ar = A @ r
        rTr = np.dot(r, r)
        denom = np.dot(r, Ar)
        if abs(denom) < 1e-30:
            return x, residuals, k
        alpha = rTr / denom

        x = x + alpha * r

    return x, residuals, maxiter

def minimal_residuals(A, b, x0=None, tol=1e-10, maxiter=5000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    residuals = []
    for k in range(maxiter):
        Ax = A @ x
        r = b - Ax
        res_norm = np.linalg.norm(Ax - b)
        residuals.append(res_norm)
        if res_norm < tol:
            return x, residuals, k

        Ar = A @ r
        numerator = np.dot(r, Ar)
        denom = np.dot(Ar, Ar)
        if abs(denom) < 1e-30:
            return x, residuals, k
        alpha = numerator / denom

        x = x + alpha * r

    return x, residuals, maxiter

def conjugate_gradients(A, b, x0=None, tol=1e-10, maxiter=5000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    residuals = []
    Ax = A @ x
    r = b - Ax
    res_norm = np.linalg.norm(Ax - b)
    residuals.append(res_norm)
    if res_norm < tol:
        return x, residuals, 0

    p = r.copy()
    rsold = np.dot(r, r)

    for k in range(1, maxiter + 1):
        Ap = A @ p
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            return x, residuals, k-1
        alpha = rsold / pAp

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = np.dot(r, r)
        Ax = A @ x
        res_norm = np.linalg.norm(Ax - b)
        residuals.append(res_norm)

        if np.sqrt(rsnew) < tol:
            return x, residuals, k

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, residuals, maxiter

def main():
    n = 10
    A = build_A(n)
    b = build_b(n)

    print('Матрица A (n=10)')
    print(A)
    print('\nПравая часть b:')
    print(b)

    x_gauss = gauss_solve(A, b)
    r_gauss = np.linalg.norm(A @ x_gauss - b)
    print('\n1) Решение методом Гаусса с выбором главного элемента:')
    print(x_gauss)
    print('Невязка ||Ax-b||_2 =', r_gauss)

    x_lu = lu_solve(A, b)
    r_lu = np.linalg.norm(A @ x_lu - b)
    print('\n2) Решение методом LU разложения:')
    print(x_lu)
    print('Невязка ||Ax-b||_2 =', r_lu)

    x0 = np.zeros(n)
    tol = 1e-10
    maxiter = 5000

    x_jacobi, res_jacobi, it_jacobi = jacobi(A, b, x0=x0, tol=tol, maxiter=maxiter)
    print('\n3) Решение методом Якоби (', it_jacobi, 'итераций):')
    print(x_jacobi)
    print('Невязка ||Ax-b||_2 =', res_jacobi[-1])

    x_gs, res_gs, it_gs = gauss_seidel(A, b, x0=x0, tol=tol, maxiter=maxiter)
    print('\n4) Решение методом Гаусса-Зейделя (', it_gs, 'итераций):')
    print(x_gs)
    print('Невязка ||Ax-b||_2 =', res_gs[-1])

    omega = 1.1
    x_sor, res_sor, it_sor = sor(A, b, omega=omega, x0=x0, tol=tol, maxiter=maxiter)
    print('\n5) Решение методом верхней реоаксации (', it_sor, 'итераций,', omega, 'итерационный параметр):')
    print(x_sor)
    print('Невязка ||Ax-b||_2 =', res_sor[-1])

    x_gr, res_gr, it_gr = gradient_descent(A, b, x0=x0, tol=tol, maxiter=maxiter)
    print('\n6) Решение методом градиентного спуска (', it_gr, 'итераций):')
    print(x_gr)
    print('Невязка ||Ax-b||_2 =', res_gr[-1])

    x_min, res_min, it_min = minimal_residuals(A, b, x0=x0, tol=tol, maxiter=maxiter)
    print('\n7) Решение методом минимальных невязок (', it_min, 'итераций):')
    print(x_min)
    print('Невязка ||Ax-b||_2 =', res_min[-1])

    x_con, res_con, it_con = conjugate_gradients(A, b, x0=x0, tol=tol, maxiter=maxiter)
    print('\n8) Решение методом сопряженных градиентов (', it_con, 'итераций):')
    print(x_con)
    print('Невязка ||Ax-b||_2 =', res_con[-1])

    plt.figure(figsize=(8,6))
    if len(res_gs) > 0:
        plt.semilogy(range(1, len(res_gs)+1), res_gs, label='Гаусса-Зейделя')
    if len(res_sor) > 0:
        plt.semilogy(range(1, len(res_sor)+1), res_sor, label=f'Верхней релаксации (omega={omega})')
    if len(res_gr) > 0:
        plt.semilogy(range(1, len(res_gr)+1), res_gr, label=f'Градиентного спуска')
    if len(res_con) > 0:
        plt.semilogy(range(1, len(res_con)+1), res_con, label=f'Сопряженных градиентов')
    if len(res_min) > 0:
        plt.semilogy(range(1, len(res_min)+1), res_min, label=f'Минимальных невязок')
    plt.xlabel('Итерация')
    plt.ylabel('||Ax - b||_2')
    plt.title('Зависимость невязки от итерации')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('residuals.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
