import math

# -------------------------------
# 1. Define parameters & helpers
# -------------------------------

# Bandwidth-like parameter for the Gaussian exponent
alpha = 1.0  # You can tune this

# Suppose each action a has a different linear function mu(a, s) = c[a] + d[a]*s
# Here we just illustrate with two actions, a=0 and a=1.
c = {0: 0.0, 1: 1.0}  # Intercepts
d = {0: 0.3, 1: 0.4}  # Slopes


def mu(a, s):
    """
    Returns the 'preferred' next-state mean mu_a(s) for a given action a.
    """
    return c[a] + d[a] * s


# Normalizing constant for the Gaussian kernel in s',
# integral_{-∞ to ∞} exp(-alpha*(x - mean)^2) dx = sqrt(pi / alpha).
# We'll store this for quick lookup:
normalizing_const = math.sqrt(math.pi / alpha)


# --------------------------------------
# 2. Define the transition PDF function
# --------------------------------------

def P_h(s_prime, s, a):
    """
    Returns the probability density P_h(s' | s, a),
    where s' is real-valued and s is real-valued,
    and a is a discrete action (0 or 1 here).

    We use a Gaussian with mean mu(a, s) and variance = 1/(2 alpha).
    """
    mean = mu(a, s)
    # The exponent part:
    exponent = -alpha * (s_prime - mean) ** 2
    # Numerator = exp(exponent):
    numerator = math.exp(exponent)
    # Denominator = normalizing_const = sqrt(pi/alpha):
    # So the PDF at s_prime is numerator / normalizing_const
    return numerator / normalizing_const


# --------------------------------------------------------
# 3. Quick demonstration of calling P_h for some examples
# --------------------------------------------------------
if __name__ == "__main__":
    # Let's evaluate P_h(s' | s, a) for some inputs
    test_states = [0.0, 0.5, 1.0, -0.5, -1.0]
    test_sprimes = [0.0, 1.0, 2.0]
    actions = [0, 1]

    print("Gaussian RBF–based transition function P_h(s' | s, a):\n")
    for s in test_states:
        for a in actions:
            print(f"--- For s={s}, a={a}, mean = mu(a,s)={mu(a, s):.3f} ---")
            for s_prime in test_sprimes:
                val = P_h(s_prime, s, a)
                print(f"  P_h(s'={s_prime} | s={s}, a={a}) = {val:.6f}")
            print()
