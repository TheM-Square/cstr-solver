"""
CSTR Equilibrium Solver
Finds steady-state conversion in a CSTR for a first-order irreversible reaction:
    A to B,  -rA = k*CA

Design equation (mole balance)-->
    tau = (CA0 - CA) / (k * CA)
    => CA = CA0 / (1 + k * tau)
    => X = k * tau / (1 + k * tau)

We solve for X given CA0, tau, and k using three root-finding methods.
Residual: f(X) = X - k*tau*(1 - X)  =>  f(X) = 0 at steady state.
"""

import numpy as np #we can also code this without importing numpy, but it will be more verbose and less efficient.  Numpy provides convenient array operations and math functions that make the code cleaner and faster.
import matplotlib.pyplot as plt


# ---------- problem parameters ----------
CA0 = 2.0    # mol/L  (feed concentration)
k   = 0.8    # 1/min  (rate constant)
tau = 3.0    # min    (residence time)

# analytical answer (for verification)
X_analytical = k * tau / (1 + k * tau)

def residual(X):
    """f(X) = X - k*tau*(1 - X).  Root = steady-state conversion."""
    return X - k * tau * (1.0 - X)


# ============================================================
# 1. BISECTION METHOD
# ============================================================
def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Find root of f in [a, b] by repeatedly halving the interval.
    Requires f(a) and f(b) to have opposite signs.
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    history = []
    for i in range(max_iter):
        mid = (a + b) / 2.0
        history.append(mid)

        if abs(f(mid)) < tol or (b - a) / 2.0 < tol:
            return mid, i + 1, history

        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid

    return (a + b) / 2.0, max_iter, history


# ============================================================
# 2. NEWTON-RAPHSON METHOD
# ============================================================
def newton_raphson(f, df, x0, tol=1e-6, max_iter=50):
    """
    Iterative root-finding using tangent line approximation.
    x_{n+1} = x_n - f(x_n) / f'(x_n)
    """
    x = x0
    history = [x]

    for i in range(max_iter):
        fx  = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            raise ZeroDivisionError("Derivative too small — possible flat region.")

        x_new = x - fx / dfx
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, i + 1, history

        x = x_new

    return x, max_iter, history


def residual_deriv(X):
    """Analytical derivative of residual wrt X===>  d/dX [X - k*tau*(1-X)] = 1 + k*tau"""
    return 1.0 + k * tau


# ============================================================
# 3. SECANT METHOD
# ============================================================
def secant(f, x0, x1, tol=1e-6, max_iter=50):
    """
    Like Newton-Raphson but uses a finite-difference approximation of the derivative.
    Does NOT require an analytical derivative.
    """
    history = [x0, x1]

    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)

        if abs(f1 - f0) < 1e-12:
            raise ZeroDivisionError("Division by near-zero difference.")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)

        if abs(x2 - x1) < tol:
            return x2, i + 1, history

        x0, x1 = x1, x2

    return x1, max_iter, history


# ============================================================
# RUNNING ALL THREE & COMPARINGGG
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  CSTR Steady-State Conversion Solver")
    print(f"  CA0={CA0} mol/L  |  k={k} 1/min  |  tau={tau} min")
    print("=" * 55)
    print(f"  Analytical answer: X = {X_analytical:.6f}\n")

    # Bisection
    X_bis, n_bis, hist_bis = bisection(residual, 0.0, 0.9999)
    print(f"  Bisection        :  X = {X_bis:.6f}  ({n_bis} iterations)")

    # Newton-Raphson
    X_nr, n_nr, hist_nr = newton_raphson(residual, residual_deriv, x0=0.5)
    print(f"  Newton-Raphson   :  X = {X_nr:.6f}  ({n_nr} iterations)")

    # Secant
    X_sec, n_sec, hist_sec = secant(residual, 0.3, 0.6)
    print(f"  Secant           :  X = {X_sec:.6f}  ({n_sec} iterations)")

    # ---- convergence plot ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(hist_bis)), [abs(x - X_analytical) for x in hist_bis],
            marker='o', label='Bisection', linewidth=1.5)
    ax.plot(range(len(hist_nr)),  [abs(x - X_analytical) for x in hist_nr],
            marker='s', label='Newton-Raphson', linewidth=1.5)
    ax.plot(range(len(hist_sec)), [abs(x - X_analytical) for x in hist_sec],
            marker='^', label='Secant', linewidth=1.5)

    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|Error|  (log scale)')
    ax.set_title('Convergence Comparison — CSTR Root-Finding')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cstr_convergence.png', dpi=120)
    print("\n  Convergence plot saved → cstr_convergence.png")
