# CSTR Equilibrium Solver

Root-finding methods applied to a steady-state CSTR mole balance. Implements Bisection, Newton-Raphson, and Secant methods from scratch in Python and compares their convergence behaviour.

![CSTR Solver Diagram](/diagram.svg)

---

## Problem Statement

For a first-order irreversible reaction **A → B** in a CSTR, the steady-state mole balance gives:

$$\tau = \frac{C_{A0} - C_A}{k \cdot C_A} \quad \Longrightarrow \quad X = \frac{k\tau}{1 + k\tau}$$

We frame this as a root-finding problem on the residual:

$$f(X) = X - k\tau(1 - X) = 0$$

and solve it three ways.

---

## Methods

| Method | Convergence Order | Requires Derivative | Iterations (this case) |
|---|---|---|---|
| Bisection | Linear (1st) | No | 20 |
| Newton-Raphson | Quadratic (2nd) | Yes (analytical) | 2 |
| Secant | Superlinear (~1.6) | No (finite diff) | 2–3 |

### Bisection
Repeatedly halves the bracket `[a, b]` where `f(a)` and `f(b)` have opposite signs. Slow but guaranteed to converge.

### Newton-Raphson
Uses the tangent line at the current estimate:
```
x_{n+1} = x_n - f(x_n) / f'(x_n)
```
Converges very fast near the root, but needs an analytical derivative.

### Secant
Same idea as Newton-Raphson but approximates the derivative using two previous points — no closed-form derivative needed.

---

## Results

```
Analytical answer:  X = 0.705882
Bisection        :  X = 0.705882  (20 iterations)
Newton-Raphson   :  X = 0.705882  (2  iterations)
Secant           :  X = 0.705882  (2  iterations)
```

All three converge to the same answer. The convergence plot shows Newton-Raphson and Secant dropping to machine precision in 2–3 steps, while Bisection takes 20.

---

## Parameters (editable at top of script)

```python
CA0 = 2.0    # mol/L  — feed concentration
k   = 0.8    # 1/min  — first-order rate constant
tau = 3.0    # min    — residence time
```

---

## Usage

```bash
pip install numpy matplotlib
python cstr_solvers.py
```

Outputs a convergence comparison plot (`cstr_convergence.png`).
![/CSTR PLOT.png]

---

## File Structure


---

## Concepts Covered

- CSTR design equation and mole balance
- Root-finding: Bisection, Newton-Raphson, Secant
- Convergence order and error analysis
- Visualising iterative solver behaviour
