import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sin(x * x)

def df1(x):
    return 2 * x * np.cos(x * x)

def f2(x):
    return np.cos(np.sin(x))

def df2(x):
    return -np.sin(np.sin(x)) * np.cos(x)

def f3(x):
    return np.exp(np.sin(np.cos(x)))

def df3(x):
    return np.exp(np.sin(np.cos(x))) * (np.cos(np.cos(x)) * (-np.sin(x)))

def f4(x):
    return np.log(x + 3)

def df4(x):
    return 1.0 / (x + 3)

def f5(x):
    return np.sqrt(x + 3)

def df5(x):
    return 0.5 * (x + 3) ** (-0.5)

functions = [
    (f1, df1, "sin(x^2)"),
    (f2, df2, "cos(sin(x))"),
    (f3, df3, "exp(sin(cos(x)))"),
    (f4, df4, "log(x+3)"),
    (f5, df5, "sqrt(x+3)")
]

def diff_method_1(f, x, h):
    return (f(x + h) - f(x)) / h

def diff_method_2(f, x, h):
    return (f(x) - f(x - h)) / h

def diff_method_3(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def diff_method_4(f, x, h):
    term1 = (f(x + h) - f(x - h)) / (2 * h)
    term2 = (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    return (4.0 / 3.0) * term1 - (1.0 / 3.0) * term2

def diff_method_5(f, x, h):
    t1 = (f(x + h) - f(x - h)) / (2 * h)
    t2 = (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    t3 = (f(x + 3 * h) - f(x - 3 * h)) / (6 * h)
    return (3.0 / 2.0) * t1 - (3.0 / 5.0) * t2 + (1.0 / 10.0) * t3

methods = [diff_method_1, diff_method_2, diff_method_3, diff_method_4, diff_method_5]
method_labels = ['(1)', '(2)', '(3)', '(4)', '(5)']

def abs_error_plot(f, df, title):
    x0 = 10.123
    h = np.logspace(-20, 0, 1000, base=2)

    plt.figure(figsize=(15, 10))
    plt.title(title)
    plt.grid(which='major')
    plt.grid(which='minor', linestyle='--')
    plt.minorticks_on()

    true_val = df(x0)

    for label, method in zip(method_labels, methods):
        approx = method(f, x0, h)
        abs_err = np.abs(approx - true_val)
        plt.plot(h, abs_err, label=label)

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('h')
    plt.ylabel('absolute error')
    plt.legend()
    plt.tight_layout()
    plt.show()

for f, df, name in functions:
    abs_error_plot(f, df, name)
