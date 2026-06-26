"""
ccqm_form_factors.py

Reference CCQM code for pseudoscalar-initial transitions:

    P_i(0-) -> P_f(0-):
        vector current  gamma^mu                 -> F_plus, F_minus
        scalar current  1                        -> F_S
        tensor current  sigma^{mu nu} q_nu       -> F_T

    P_i(0-) -> V_f(1-):
        V-A current     gamma^mu(1-gamma5)       -> A0, A_plus, A_minus, V
        tensor current  sigma^{mu nu}q_nu(1+g5)  -> a0, a_plus, g

This is a transparent reference implementation.  It uses:
    metric g = diag(+1,-1,-1,-1)
    sigma^{mu nu} = i/2 [gamma^mu, gamma^nu]
    gamma5 = i gamma0 gamma1 gamma2 gamma3

Important:
    Tensor-current conventions differ between papers.  For P->V rare amplitudes
    many formulas use i sigma^{mu nu} q_nu(1+gamma5).  The function compute_pv
    therefore has pv_tensor_i_prefactor=True by default so that the projected
    tensor form factors are real in this convention.
"""

from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np

# ---------------------------------------------------------------------
# Gamma matrices and Lorentz utilities
# ---------------------------------------------------------------------

G = np.diag([1.0, -1.0, -1.0, -1.0])      # g^{mu nu}=g_{mu nu}
SIGN = np.array([1.0, -1.0, -1.0, -1.0])
I4 = np.eye(4, dtype=complex)


def gamma_matrices():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    zero = np.zeros((2, 2), dtype=complex)
    I2 = np.eye(2, dtype=complex)

    gamma0 = np.block([[I2, zero], [zero, -I2]])

    def spatial(sigma):
        return np.block([[zero, sigma], [-sigma, zero]])

    gamma = [gamma0, spatial(sx), spatial(sy), spatial(sz)]
    gamma5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    return gamma, gamma5


GAMMA, GAMMA5 = gamma_matrices()


def sigma_matrix(mu: int, nu: int) -> np.ndarray:
    return 0.5j * (GAMMA[mu] @ GAMMA[nu] - GAMMA[nu] @ GAMMA[mu])


SIGMA = [[sigma_matrix(mu, nu) for nu in range(4)] for mu in range(4)]


def dot(a, b):
    """Minkowski dot product a.b with metric (+---)."""
    return np.dot(np.asarray(a), SIGN * np.asarray(b))


def lower(v):
    """Lower a contravariant four-vector: v_mu = g_{mu nu} v^nu."""
    return SIGN * np.asarray(v)


def levi_civita4():
    eps = np.zeros((4, 4, 4, 4), dtype=float)
    for p in itertools.permutations(range(4)):
        inv = 0
        for i in range(4):
            for j in range(i + 1, 4):
                inv += p[i] > p[j]
        eps[p] = -1.0 if inv % 2 else 1.0
    return eps


EPS = -levi_civita4()  # CCQM papers use epsilon^0123 = -1


def sigma_q(mu: int, q) -> np.ndarray:
    """sigma^{mu nu} q_nu."""
    q_lower = lower(q)
    out = np.zeros((4, 4), dtype=complex)
    for nu in range(4):
        out += SIGMA[mu][nu] * q_lower[nu]
    return out


# ---------------------------------------------------------------------
# Model input
# ---------------------------------------------------------------------

@dataclass
class CCQMConfig:
    Mi: float
    Mf: float

    # quark masses: q1 initial active, q2 final active, q3 spectator
    m1: float
    m2: float
    m3: float

    # size parameters
    Lambda_i: float
    Lambda_f: float

    # meson-quark couplings from compositeness condition
    g_i: float = 1.0
    g_f: float = 1.0

    # infrared cutoff lambda
    lambda_ir: float = 0.181

    Nc: int = 3
    n_quad: int = 10


def kinematics(Mi: float, Mf: float, q2: float):
    """
    Rest-frame kinematics:
        p1 = (Mi,0,0,0)
        p2 = (E2,0,0,|p|)
    Valid for 0 <= q2 <= (Mi-Mf)^2.
    """
    q2max = (Mi - Mf) ** 2
    if q2 < -1e-12 or q2 > q2max + 1e-12:
        raise ValueError(
            f"q2={q2} outside physical range [0,{q2max}] for this rest-frame setup"
        )

    E2 = (Mi * Mi + Mf * Mf - q2) / (2.0 * Mi)
    pz2 = E2 * E2 - Mf * Mf
    pz = np.sqrt(max(pz2, 0.0))

    p1 = np.array([Mi, 0.0, 0.0, 0.0])
    p2 = np.array([E2, 0.0, 0.0, pz])
    return p1, p2


def gauss_nodes(a: float, b: float, n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


# ---------------------------------------------------------------------
# Gaussian moments and generic trace averaging
# ---------------------------------------------------------------------

def moment_lower(selected, K, C):
    """
    Moment of lower components of affine vectors.

    selected = [(X, lam), ...]
    means component lam of x_mu = g_{mu nu}(k+X)^nu.

    K^mu = <k^mu>
    C^{mu nu} = <(k-K)^mu (k-K)^nu> = -g^{mu nu}/(2A)
    """
    n = len(selected)

    if n == 0:
        return 1.0 + 0.0j

    bars = []
    for X, lam in selected:
        bars.append(SIGN[lam] * (K[lam] + X[lam]))

    def C_lower_lower(i, j):
        _, li = selected[i]
        _, lj = selected[j]
        return SIGN[li] * SIGN[lj] * C[li, lj]

    if n == 1:
        return bars[0]

    if n == 2:
        return bars[0] * bars[1] + C_lower_lower(0, 1)

    if n == 3:
        return (
            bars[0] * bars[1] * bars[2]
            + C_lower_lower(0, 1) * bars[2]
            + C_lower_lower(0, 2) * bars[1]
            + C_lower_lower(1, 2) * bars[0]
        )

    raise NotImplementedError("Only up to three slashed propagator momenta are expected.")


def trace_average(sequence, masses, offsets, K, C):
    """
    Gaussian average of a Dirac trace.

    sequence contains matrices and propagator labels:
        'N1' means m1 + slash(k+p1)
        'N2' means m2 + slash(k+p2)
        'N3' means m3 + slash(k)

    The slash convention is slash(v)=gamma^mu v_mu.
    """
    mass = {"N1": masses[0], "N2": masses[1], "N3": masses[2]}
    offset = {"N1": offsets[0], "N2": offsets[1], "N3": offsets[2]}

    total = 0.0 + 0.0j

    def rec(i, M, selected):
        nonlocal total

        if i == len(sequence):
            total += np.trace(M) * moment_lower(selected, K, C)
            return

        item = sequence[i]

        if isinstance(item, str):
            # mass branch
            rec(i + 1, M @ (mass[item] * I4), selected)

            # slash branch: gamma^lambda (k+X)_lambda
            for lam in range(4):
                rec(i + 1, M @ GAMMA[lam], selected + [(offset[item], lam)])
        else:
            rec(i + 1, M @ item, selected)

    rec(0, I4, [])
    return total


# ---------------------------------------------------------------------
# Schwinger/simplex integration
# ---------------------------------------------------------------------

def integrate_amplitude(cfg: CCQMConfig, q2: float, evaluator, shape):
    """
    Master integral:
      Nc g_i g_f/(4 pi^2)
      int_0^{1/lambda^2} dt t^2 int d alpha_1 d alpha_2
      exp(z-r^2/A)/A^2 * averaged_trace
    """
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)

    si = 1.0 / cfg.Lambda_i ** 2
    sf = 1.0 / cfg.Lambda_f ** 2

    w13 = cfg.m3 / (cfg.m1 + cfg.m3)
    w23 = cfg.m3 / (cfg.m2 + cfg.m3)

    tmax = 1.0 / cfg.lambda_ir ** 2
    xs, ws = gauss_nodes(0.0, 1.0, cfg.n_quad)
    ts, wt = gauss_nodes(0.0, tmax, cfg.n_quad)

    out = np.zeros(shape, dtype=complex)

    for t, wt_i in zip(ts, wt):
        A = si + sf + t
        C = -G / (2.0 * A)  # C^{mu nu}

        for alpha1, wa1 in zip(xs, ws):
            for u, wu in zip(xs, ws):
                alpha2 = (1.0 - alpha1) * u
                alpha3 = 1.0 - alpha1 - alpha2
                jac = 1.0 - alpha1

                rho1 = si * w13 + t * alpha1
                rho2 = sf * w23 + t * alpha2

                r = rho1 * p1 + rho2 * p2

                z = (
                    si * w13 * w13 * cfg.Mi ** 2
                    + sf * w23 * w23 * cfg.Mf ** 2
                    + t * alpha1 * (cfg.Mi ** 2 - cfg.m1 ** 2)
                    + t * alpha2 * (cfg.Mf ** 2 - cfg.m2 ** 2)
                    - t * alpha3 * cfg.m3 ** 2
                )

                W = z - dot(r, r) / A
                K = -r / A

                weight = wt_i * wa1 * wu * jac * t * t * np.exp(W) / (A * A)
                out += weight * evaluator(p1, p2, p1 - p2, K, C)

    prefactor = cfg.Nc * cfg.g_i * cfg.g_f / (4.0 * np.pi * np.pi)
    return prefactor * out


# ---------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------

def project_pp_vector(M, p1, p2):
    P = p1 + p2
    q = p1 - p2

    P2 = dot(P, P)
    q2 = dot(q, q)
    D = dot(P, q)

    PdM = dot(P, M)
    qdM = dot(q, M)

    Delta = P2 * q2 - D * D

    Fp = (PdM * q2 - qdM * D) / Delta
    Fm = (qdM * P2 - PdM * D) / Delta
    return Fp, Fm


def project_pp_tensor(MT, p1, p2, Mi, Mf):
    """
    MT^mu = i/(Mi+Mf) [ q^2 P^mu - (P.q)q^mu ] F_T.

    We project MT/i onto P^mu and q^mu:
        MT/i = A P + B q
        B = -(P.q)/(Mi+Mf) F_T
    This remains stable at q2=0.
    """
    P = p1 + p2
    q = p1 - p2
    D = dot(P, q)

    R = MT / (1j)
    _, Bcoef = project_pp_vector(R, p1, p2)
    return -(Mi + Mf) * Bcoef / D


def inner_rank2(A, B, p2, Mf):
    """
    Inner product for P->V projections:
        A^{mu nu} g_{mu mu'} Pi_{nu rho} B^{mu' rho}
    with Pi_{nu rho} = -g_{nu rho} + p2_nu p2_rho/Mf^2.
    """
    p2_lower = lower(p2)
    Pi_lower = -G + np.outer(p2_lower, p2_lower) / (Mf * Mf)

    s = 0.0 + 0.0j
    for mu in range(4):
        for mup in range(4):
            g_mu = G[mu, mup]
            if g_mu == 0:
                continue
            for nu in range(4):
                for rho in range(4):
                    s += A[mu, nu] * g_mu * Pi_lower[nu, rho] * B[mup, rho]
    return s


def basis_pv_va(p1, p2, Mi, Mf):
    P = p1 + p2
    q = p1 - p2
    D = dot(P, q)
    denom = Mi + Mf

    B0 = -D / denom * G.astype(complex)
    Bp = np.outer(P, P) / denom
    Bm = np.outer(q, P) / denom

    P_lower = lower(P)
    q_lower = lower(q)

    BV = np.zeros((4, 4), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            s = 0.0
            for a in range(4):
                for b in range(4):
                    s += EPS[mu, nu, a, b] * P_lower[a] * q_lower[b]
            BV[mu, nu] = 1j * s / denom

    return [B0, Bp, Bm, BV]


def basis_pv_tensor(p1, p2):
    P = p1 + p2
    q = p1 - p2
    D = dot(P, q)
    q2 = dot(q, q)

    if abs(q2) < 1e-12:
        raise ValueError("P->V tensor basis contains 1/q2. Use q2>0 and extrapolate to q2=0.")

    C0 = -(G - np.outer(q, q) / q2) * D
    Cp = np.outer(P, P) - np.outer(q, P) * D / q2

    P_lower = lower(P)
    q_lower = lower(q)

    Cg = np.zeros((4, 4), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            s = 0.0
            for a in range(4):
                for b in range(4):
                    s += EPS[mu, nu, a, b] * P_lower[a] * q_lower[b]
            Cg[mu, nu] = 1j * s

    return [C0, Cp, Cg]


def solve_projection(M, bases, p2, Mf):
    n = len(bases)
    Gmat = np.zeros((n, n), dtype=complex)
    Y = np.zeros(n, dtype=complex)

    for a in range(n):
        Y[a] = inner_rank2(bases[a], M, p2, Mf)
        for b in range(n):
            Gmat[a, b] = inner_rank2(bases[a], bases[b], p2, Mf)

    return np.linalg.solve(Gmat, Y)


# ---------------------------------------------------------------------
# Public API: P -> P
# ---------------------------------------------------------------------

def compute_pp(cfg: CCQMConfig, q2: float):
    """
    Calculate P_i(0-) -> P_f(0-) form factors:
        F_plus, F_minus, F_S, F_T
    """
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)

    offsets = (p1, p2, np.zeros(4))       # N1, N2, N3
    masses = (cfg.m1, cfg.m2, cfg.m3)

    def eval_vector(p1, p2, q, K, C):
        out = np.zeros(4, dtype=complex)
        for mu in range(4):
            seq = [GAMMA[mu], "N1", GAMMA5, "N3", GAMMA5, "N2"]
            out[mu] = trace_average(seq, masses, offsets, K, C)
        return out

    def eval_scalar(p1, p2, q, K, C):
        seq = ["N1", GAMMA5, "N3", GAMMA5, "N2"]
        return np.array(trace_average(seq, masses, offsets, K, C))

    def eval_tensor(p1, p2, q, K, C):
        out = np.zeros(4, dtype=complex)
        for mu in range(4):
            seq = [sigma_q(mu, q), "N1", GAMMA5, "N3", GAMMA5, "N2"]
            out[mu] = trace_average(seq, masses, offsets, K, C)
        return out

    M_vector = integrate_amplitude(cfg, q2, eval_vector, (4,))
    M_scalar = integrate_amplitude(cfg, q2, eval_scalar, ())
    M_tensor = integrate_amplitude(cfg, q2, eval_tensor, (4,))

    F_plus, F_minus = project_pp_vector(M_vector, p1, p2)
    F_S = M_scalar / (cfg.Mi + cfg.Mf)
    F_T = project_pp_tensor(M_tensor, p1, p2, cfg.Mi, cfg.Mf)

    return {
        "F_plus": F_plus,
        "F_minus": F_minus,
        "F_S": F_S,
        "F_T": F_T,
        "M_vector": M_vector,
        "M_scalar": M_scalar,
        "M_tensor": M_tensor,
    }


# ---------------------------------------------------------------------
# Public API: P -> V
# ---------------------------------------------------------------------

def compute_pv(cfg: CCQMConfig, q2: float, pv_tensor_i_prefactor: bool = True, va_current_factor: float = 1.0):
    """
    Calculate P_i(0-) -> V_f(1-) form factors:
        A0, A_plus, A_minus, V
        a0, a_plus, g

    pv_tensor_i_prefactor:
        True  -> current is i sigma^{mu nu} q_nu (1+gamma5), useful for rare-amplitude convention.
        False -> current is   sigma^{mu nu} q_nu (1+gamma5).

    va_current_factor:
        Multiplicative convention factor applied only to the P->V vector/axial amplitude
        before projecting A0, A_plus, A_minus, V.  Default is 1.0.
        Use 2.0 only when matching conventions that use twice this V-A normalization.
    """
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)

    offsets = (p1, p2, np.zeros(4))       # N1, N2, N3
    masses = (cfg.m1, cfg.m2, cfg.m3)

    def eval_va(p1, p2, q, K, C):
        out = np.zeros((4, 4), dtype=complex)
        for mu in range(4):
            current = GAMMA[mu] @ (I4 - GAMMA5)
            for nu in range(4):
                seq = [current, "N1", GAMMA5, "N3", GAMMA[nu], "N2"]
                out[mu, nu] = trace_average(seq, masses, offsets, K, C)
        return out

    def eval_tensor(p1, p2, q, K, C):
        out = np.zeros((4, 4), dtype=complex)
        for mu in range(4):
            current = sigma_q(mu, q) @ (I4 + GAMMA5)
            if pv_tensor_i_prefactor:
                current = 1j * current
            for nu in range(4):
                seq = [current, "N1", GAMMA5, "N3", GAMMA[nu], "N2"]
                out[mu, nu] = trace_average(seq, masses, offsets, K, C)
        return out

    M_VA = va_current_factor * integrate_amplitude(cfg, q2, eval_va, (4, 4))
    M_T = integrate_amplitude(cfg, q2, eval_tensor, (4, 4))

    A0, A_plus, A_minus, V = solve_projection(
        M_VA, basis_pv_va(p1, p2, cfg.Mi, cfg.Mf), p2, cfg.Mf
    )

    a0, a_plus, g = solve_projection(
        M_T, basis_pv_tensor(p1, p2), p2, cfg.Mf
    )

    return {
        "A0": A0,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "V": V,
        "a0": a0,
        "a_plus": a_plus,
        "g": g,
        "M_VA": M_VA,
        "M_T": M_T,
    }


def real_if_close_dict(d, tol=1000):
    """Convenience display helper."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = np.real_if_close(v, tol=tol)
        else:
            out[k] = np.real_if_close(v, tol=tol)
    return out




# ---------------------------------------------------------------------
# Meson-quark couplings from compositeness condition Z_H = 0
# ---------------------------------------------------------------------

@dataclass
class MesonConfig:
    """
    Input for a single pseudoscalar/vector meson H(q1 qbar2).

    kind:
        "P" for pseudoscalar, Gamma_H = gamma5
        "V" for vector,       Gamma_H = gamma^mu

    The meson-quark coupling is fixed by
        Z_H = 1 - Nc*g_H^2/(4*pi^2) * Pi_H'(M_H^2) = 0,
    i.e.
        g_H = sqrt(4*pi^2/(Nc*Pi_H'(M_H^2))).

    The function below evaluates Pi_H' numerically by a central finite
    difference of the two-point mass operator.  It is meant as a transparent
    reference implementation consistent with the Schwinger/Gaussian machinery
    used for the decay constants and form factors in this file.
    """
    name: str
    kind: str
    M: float
    m1: float
    m2: float
    Lambda: float
    lambda_ir: float = 0.181
    Nc: int = 3
    n_quad: int = 64


def _kind_alias(kind: str) -> str:
    k = kind.upper().strip()
    if k in ("P", "PS", "PSEUDOSCALAR"):
        return "P"
    if k in ("V", "VECTOR"):
        return "V"
    raise ValueError("Only kind='P' and kind='V' are implemented for meson couplings.")


def _two_point_mass_operator(mc: MesonConfig, p2_value: float, vertex_power: int = 2):
    """
    Reduced two-point mass operator Pi_H(p^2), without the external g_H^2
    and without the overall Nc/(4*pi^2) compositeness prefactor.

    The Gaussian factor is Phi_H^vertex_power.  For the mass operator in the
    compositeness condition one normally needs Phi_H^2, so vertex_power=2.
    """
    kind = _kind_alias(mc.kind)
    if p2_value <= 0.0:
        raise ValueError("p2_value must be positive for the rest-frame projection.")

    p = np.array([np.sqrt(p2_value), 0.0, 0.0, 0.0])
    w1 = mc.m1 / (mc.m1 + mc.m2)
    w2 = mc.m2 / (mc.m1 + mc.m2)

    # One vertex gives exp(k^2/Lambda^2); the mass operator contains Phi^2.
    s = float(vertex_power) / (mc.Lambda * mc.Lambda)
    tmax = 1.0 / (mc.lambda_ir * mc.lambda_ir)
    ts, wt = gauss_nodes(0.0, tmax, mc.n_quad)
    us, wu = gauss_nodes(0.0, 1.0, mc.n_quad)

    masses = (mc.m1, mc.m2, 0.0)

    if kind == "P":
        total = 0.0 + 0.0j
    else:
        total = np.zeros((4, 4), dtype=complex)

    for t, w_t in zip(ts, wt):
        A = s + t
        C = -G / (2.0 * A)

        for u, w_u in zip(us, wu):
            beta1 = t * u
            beta2 = t * (1.0 - u)

            X1 = w1 * p
            X2 = -w2 * p
            offsets = (X1, X2, np.zeros(4))

            r = beta1 * X1 + beta2 * X2
            z = beta1 * (dot(X1, X1) - mc.m1 * mc.m1) + beta2 * (dot(X2, X2) - mc.m2 * mc.m2)
            W = z - dot(r, r) / A
            K = -r / A

            weight = w_t * w_u * t * np.exp(W) / (A * A)

            if kind == "P":
                seq = [GAMMA5, "N1", GAMMA5, "N2"]
                total += weight * trace_average(seq, masses, offsets, K, C)
            else:
                for mu in range(4):
                    for nu in range(4):
                        seq = [GAMMA[mu], "N1", GAMMA[nu], "N2"]
                        total[mu, nu] += weight * trace_average(seq, masses, offsets, K, C)

    if kind == "P":
        return total

    # Extract the transverse coefficient: (1/3)(g_{mu nu}-p_mu p_nu/p^2) Pi^{mu nu}.
    p_lower = lower(p)
    projector = G - np.outer(p_lower, p_lower) / p2_value
    return np.sum(projector * total) / 3.0


def compute_coupling_constant(mc: MesonConfig, derivative_step: float = 1e-4, use_abs_if_negative: bool = True):
    """
    Compute the meson-quark coupling g_H from the compositeness condition.

    Returns a dictionary containing g_H and the reduced derivative Pi'_H(M_H^2).
    If a sign convention makes Pi'_H negative, use_abs_if_negative=True returns
    sqrt(abs(4*pi^2/(Nc*Pi'))) and marks the result with used_abs=True.
    """
    s0 = mc.M * mc.M
    h = derivative_step * max(1.0, s0)

    if s0 - h > 0.0:
        pi_plus = _two_point_mass_operator(mc, s0 + h)
        pi_minus = _two_point_mass_operator(mc, s0 - h)
        pi_prime = (pi_plus - pi_minus) / (2.0 * h)
        stencil = "central"
    else:
        pi_0 = _two_point_mass_operator(mc, s0)
        pi_plus = _two_point_mass_operator(mc, s0 + h)
        pi_prime = (pi_plus - pi_0) / h
        stencil = "forward"

    pi_prime = np.real_if_close(pi_prime, tol=1000)
    pi_prime_real = float(np.real(pi_prime))

    raw = 4.0 * np.pi * np.pi / (mc.Nc * pi_prime_real)
    used_abs = False
    if raw < 0.0:
        if not use_abs_if_negative:
            raise ValueError(
                "Compositeness derivative produced negative normalization. "
                "Check gamma/vertex conventions or call with use_abs_if_negative=True."
            )
        raw = abs(raw)
        used_abs = True

    gH = float(np.sqrt(raw))
    return {
        "name": mc.name,
        "kind": _kind_alias(mc.kind),
        "g": gH,
        "Pi_prime": pi_prime_real,
        "Z_check": 1.0 - mc.Nc * gH * gH * pi_prime_real / (4.0 * np.pi * np.pi),
        "used_abs": used_abs,
        "derivative_step": derivative_step,
        "stencil": stencil,
        "definition": "Z_H = 1 - Nc*g_H^2/(4*pi^2)*Pi_H'(M_H^2) = 0",
    }


def compute_couplings(input_data: dict):
    """
    Compute couplings for all entries in input_data['mesons'].

    Example:
      {"global":{"lambda_ir":0.181,"n_quad":32,"Nc":3},
       "mesons":[{"name":"Ds","kind":"P","M":1.968,"m1":1.672,"m2":0.428,"Lambda":1.75}]}
    """
    glob = input_data.get("global", {})
    lambda_ir = glob.get("lambda_ir", 0.181)
    n_quad = glob.get("n_quad", 64)
    Nc = glob.get("Nc", 3)

    out = {}
    for item in input_data.get("mesons", []):
        mc = MesonConfig(
            name=item.get("name", f"meson_{len(out)+1}"),
            kind=item["kind"],
            M=item["M"],
            m1=item["m1"],
            m2=item["m2"],
            Lambda=item["Lambda"],
            lambda_ir=item.get("lambda_ir", lambda_ir),
            Nc=item.get("Nc", Nc),
            n_quad=item.get("n_quad", n_quad),
        )
        out[mc.name] = compute_coupling_constant(
            mc,
            derivative_step=item.get("derivative_step", 1e-4),
            use_abs_if_negative=item.get("use_abs_if_negative", True),
        )
    return out


# ---------------------------------------------------------------------
# Decay constants: leptonic/nonleptonic building blocks
# ---------------------------------------------------------------------

@dataclass
class DecayConstantConfig:
    """
    Meson decay-constant input.

    kind:
        "P" for pseudoscalar:
            <0| qbar gamma^mu gamma5 q |P(p)> = i f_P p^mu

        "V" for vector:
            <0| qbar gamma^mu q |V(p,eps)> = M_V f_V eps^mu

    m1, m2 are the two constituent quark masses in GeV.
    Lambda is the meson size parameter in GeV.
    g is the meson-quark coupling from the compositeness condition.
    """
    kind: str
    M: float
    m1: float
    m2: float
    Lambda: float
    g: float = 1.0
    lambda_ir: float = 0.181
    Nc: int = 3
    n_quad: int = 64


def _decay_integral_common(dc: DecayConstantConfig, numerator):
    """
    Two-propagator Schwinger integral for decay constants.

    beta1=t*u, beta2=t*(1-u), 0<t<1/lambda_ir^2, 0<u<1.
    """
    s = 1.0 / (dc.Lambda * dc.Lambda)
    p2 = dc.M * dc.M

    w1 = dc.m1 / (dc.m1 + dc.m2)
    w2 = dc.m2 / (dc.m1 + dc.m2)

    tmax = 1.0 / (dc.lambda_ir * dc.lambda_ir)
    ts, wt = gauss_nodes(0.0, tmax, dc.n_quad)
    us, wu = gauss_nodes(0.0, 1.0, dc.n_quad)

    total = 0.0
    for t, w_t in zip(ts, wt):
        A = s + t
        for u, w_u in zip(us, wu):
            D = t * (u - w2)
            mass_mix = u * dc.m1 * dc.m1 + (1.0 - u) * dc.m2 * dc.m2
            omega = t * (u * w1 * w1 + (1.0 - u) * w2 * w2) - D * D / A
            expo = -t * mass_mix + omega * p2

            total += (
                w_t * w_u
                * t
                * np.exp(expo)
                / (A * A)
                * numerator(t, u, A, D, w1, w2, p2)
            )
    return total


def compute_decay_constant(dc: DecayConstantConfig):
    """
    Return the decay constant f_M in GeV.

    Supported:
        kind="P": pseudoscalar f_P
        kind="V": vector f_V

    These are the building blocks for:
        leptonic decays: P -> l nu uses f_P
        nonleptonic factorization: emitted P/V meson uses f_P or f_V
    """
    kind = dc.kind.upper().strip()

    if kind in ("P", "PS", "PSEUDOSCALAR"):
        def numerator_p(t, u, A, D, w1, w2, p2):
            return 4.0 * ((dc.m1 - dc.m2) * (-D / A) - (dc.m1 * w2 + dc.m2 * w1))

        J = _decay_integral_common(dc, numerator_p)
        f = -dc.Nc * dc.g / (8.0 * np.pi * np.pi) * J
        return {
            "kind": "P",
            "f": f,
            "f_GeV": f,
            "f_MeV": 1000.0 * f,
            "reduced_integral": J,
            "definition": "<0|qbar gamma^mu gamma5 q|P> = i f_P p^mu",
        }

    if kind in ("V", "VECTOR"):
        def numerator_v(t, u, A, D, w1, w2, p2):
            chi = w1 * w2 + (w1 - w2) * D / A - D * D / (A * A)
            return 4.0 * (dc.m1 * dc.m2 + 1.0 / A + chi * p2)

        J = _decay_integral_common(dc, numerator_v)
        f = dc.Nc * dc.g / (8.0 * np.pi * np.pi * dc.M) * J
        return {
            "kind": "V",
            "f": f,
            "f_GeV": f,
            "f_MeV": 1000.0 * f,
            "reduced_integral": J,
            "definition": "<0|qbar gamma^mu q|V> = M_V f_V epsilon^mu",
        }

    raise ValueError("Decay constants currently support kind='P' and kind='V'.")


def compute_transition_form_factors(cfg: CCQMConfig, final_kind: str, q2: float, **kwargs):
    """
    Unified wrapper for form factors.

    final_kind="P":
        returns F_plus, F_minus, F_S, F_T

    final_kind="V":
        returns A0, A_plus, A_minus, V, a0, a_plus, g
    """
    final_kind = final_kind.upper().strip()

    if final_kind in ("P", "PS", "PSEUDOSCALAR"):
        return compute_pp(cfg, q2)

    if final_kind in ("V", "VECTOR"):
        return compute_pv(
            cfg,
            q2,
            pv_tensor_i_prefactor=kwargs.get("pv_tensor_i_prefactor", True),
            va_current_factor=kwargs.get("va_current_factor", 1.0),
        )

    raise ValueError("final_kind must be 'P' or 'V'.")


def compute_building_blocks(input_data: dict):
    """
    Compute all requested decay constants and form factors from a dictionary.

    Input format:
    {
      "global": {"lambda_ir":0.181, "n_quad":12, "Nc":3},
      "decay_constants": [
        {"name":"Ds","kind":"P","M":1.9683,"m1":2.16,"m2":0.424,"Lambda":1.75,"g":1.9}
      ],
      "transitions": [
        {
          "name":"Bs_to_Ds", "final_kind":"P",
          "Mi":5.3669, "Mf":1.9683,
          "m1":5.09, "m2":2.16, "m3":0.424,
          "Lambda_i":2.05, "Lambda_f":1.75,
          "g_i":1.0, "g_f":1.0,
          "q2_values":[0.0, 1.0]
        }
      ]
    }
    """
    glob = input_data.get("global", {})
    lambda_ir = glob.get("lambda_ir", 0.181)
    n_quad = glob.get("n_quad", 10)
    Nc = glob.get("Nc", 3)

    results = {
        "decay_constants": {},
        "transitions": {},
        "notes": [
            "This output contains CCQM building blocks only: f_P/f_V and P->P/P->V form factors.",
            "It does not compute decay widths, branching fractions, angular observables, or Wilson-coefficient amplitudes.",
        ],
    }

    for item in input_data.get("decay_constants", []):
        dc = DecayConstantConfig(
            kind=item["kind"],
            M=item["M"],
            m1=item["m1"],
            m2=item["m2"],
            Lambda=item["Lambda"],
            g=item.get("g", 1.0),
            lambda_ir=item.get("lambda_ir", lambda_ir),
            Nc=item.get("Nc", Nc),
            n_quad=item.get("n_quad", n_quad),
        )
        results["decay_constants"][item.get("name", f"meson_{len(results['decay_constants'])+1}")] = compute_decay_constant(dc)

    for item in input_data.get("transitions", []):
        cfg = CCQMConfig(
            Mi=item["Mi"],
            Mf=item["Mf"],
            m1=item["m1"],
            m2=item["m2"],
            m3=item["m3"],
            Lambda_i=item["Lambda_i"],
            Lambda_f=item["Lambda_f"],
            g_i=item.get("g_i", 1.0),
            g_f=item.get("g_f", 1.0),
            lambda_ir=item.get("lambda_ir", lambda_ir),
            Nc=item.get("Nc", Nc),
            n_quad=item.get("n_quad", n_quad),
        )

        name = item.get("name", f"transition_{len(results['transitions'])+1}")
        final_kind = item["final_kind"]
        q2_values = item.get("q2_values", [0.0])

        transition_result = {
            "final_kind": final_kind,
            "q2_max": (cfg.Mi - cfg.Mf) ** 2,
            "va_current_factor": item.get("va_current_factor", 1.0) if str(final_kind).upper().startswith("V") else None,
            "pv_tensor_i_prefactor": item.get("pv_tensor_i_prefactor", True) if str(final_kind).upper().startswith("V") else None,
            "form_factors": {},
        }

        for q2 in q2_values:
            ff = compute_transition_form_factors(
                cfg,
                final_kind,
                float(q2),
                pv_tensor_i_prefactor=item.get("pv_tensor_i_prefactor", True),
                va_current_factor=item.get("va_current_factor", 1.0),
            )

            # Keep only user-facing form-factor keys.
            if final_kind.upper().startswith("P"):
                keys = ["F_plus", "F_minus", "F_S", "F_T"]
            else:
                keys = ["A0", "A_plus", "A_minus", "V", "a0", "a_plus", "g"]

            transition_result["form_factors"][str(q2)] = {k: ff[k] for k in keys}

        results["transitions"][name] = transition_result

    return results


# ---------------------------------------------------------------------
# High-level workflow: couplings -> decay constants -> form factors
# ---------------------------------------------------------------------

def compute_all_building_blocks(input_data: dict):
    """
    Full CCQM workflow.

    This first computes g_H for all mesons listed under input_data['mesons'],
    then uses those couplings in decay constants and transition form factors.

    Input format:
    {
      "global": {"lambda_ir": 0.181, "n_quad": 24, "Nc": 3},
      "mesons": [
        {"name":"Bs", "kind":"P", "M":5.366, "m1":5.046, "m2":0.428, "Lambda":2.05},
        {"name":"Ds", "kind":"P", "M":1.968, "m1":1.672, "m2":0.428, "Lambda":1.75}
      ],
      "decay_constants": [
        {"name":"Ds", "meson":"Ds"}
      ],
      "transitions": [
        {"name":"Bs_to_Ds", "initial":"Bs", "final":"Ds", "final_kind":"P",
         "m1":5.046, "m2":1.672, "m3":0.428, "q2_values":[0.0, 1.0]}
      ]
    }

    For transitions, m1 is the initial active quark, m2 is the final active quark,
    and m3 is the spectator.  Mi/Mf/Lambda_i/Lambda_f/g_i/g_f are filled from
    the named mesons when initial/final are supplied, but can still be overridden.
    """
    glob = input_data.get("global", {})
    lambda_ir = glob.get("lambda_ir", 0.181)
    n_quad = glob.get("n_quad", 24)
    Nc = glob.get("Nc", 3)

    meson_inputs = {m.get("name", f"meson_{i+1}"): dict(m) for i, m in enumerate(input_data.get("mesons", []))}
    couplings = compute_couplings(input_data) if meson_inputs else {}

    results = {
        "couplings": couplings,
        "decay_constants": {},
        "transitions": {},
        "notes": [
            "Couplings are computed from Z_H=0 for P and V mesons.",
            "Decay constants use the computed g_H unless a decay-constant item supplies g explicitly.",
            "Transition form factors use computed g_i and g_f unless supplied explicitly.",
        ],
    }

    for item in input_data.get("decay_constants", []):
        name = item.get("name") or item.get("meson") or f"meson_{len(results['decay_constants'])+1}"
        src_name = item.get("meson", name)
        src = meson_inputs.get(src_name, {})
        g_value = item.get("g", couplings.get(src_name, {}).get("g", 1.0))
        dc = DecayConstantConfig(
            kind=item.get("kind", src.get("kind")),
            M=item.get("M", src.get("M")),
            m1=item.get("m1", src.get("m1")),
            m2=item.get("m2", src.get("m2")),
            Lambda=item.get("Lambda", src.get("Lambda")),
            g=g_value,
            lambda_ir=item.get("lambda_ir", lambda_ir),
            Nc=item.get("Nc", Nc),
            n_quad=item.get("n_quad", n_quad),
        )
        results["decay_constants"][name] = compute_decay_constant(dc)

    for item in input_data.get("transitions", []):
        name = item.get("name", f"transition_{len(results['transitions'])+1}")
        initial = meson_inputs.get(item.get("initial", ""), {})
        final = meson_inputs.get(item.get("final", ""), {})
        initial_name = item.get("initial")
        final_name = item.get("final")

        cfg = CCQMConfig(
            Mi=item.get("Mi", initial.get("M")),
            Mf=item.get("Mf", final.get("M")),
            m1=item["m1"],
            m2=item["m2"],
            m3=item["m3"],
            Lambda_i=item.get("Lambda_i", initial.get("Lambda")),
            Lambda_f=item.get("Lambda_f", final.get("Lambda")),
            g_i=item.get("g_i", couplings.get(initial_name, {}).get("g", 1.0)),
            g_f=item.get("g_f", couplings.get(final_name, {}).get("g", 1.0)),
            lambda_ir=item.get("lambda_ir", lambda_ir),
            Nc=item.get("Nc", Nc),
            n_quad=item.get("n_quad", n_quad),
        )

        final_kind = item.get("final_kind", final.get("kind"))
        q2_values = item.get("q2_values", [0.0])
        transition_result = {
            "initial": initial_name,
            "final": final_name,
            "final_kind": final_kind,
            "q2_max": (cfg.Mi - cfg.Mf) ** 2,
            "g_i": cfg.g_i,
            "g_f": cfg.g_f,
            "form_factors": {},
        }

        for q2 in q2_values:
            ff = compute_transition_form_factors(
                cfg,
                final_kind,
                float(q2),
                pv_tensor_i_prefactor=item.get("pv_tensor_i_prefactor", True),
                va_current_factor=item.get("va_current_factor", 1.0),
            )
            if str(final_kind).upper().startswith("P"):
                keys = ["F_plus", "F_minus", "F_S", "F_T"]
            else:
                keys = ["A0", "A_plus", "A_minus", "V", "a0", "a_plus", "g"]
            transition_result["form_factors"][str(q2)] = {k: ff[k] for k in keys}

        results["transitions"][name] = transition_result

    return results

# =====================================================================
# Generalized P(initial) -> P/V(final) process layer
# =====================================================================
# This section turns the building blocks above into a process-oriented API.
# Scope: pseudoscalar initial meson P_i(0-) and final P_f(0-) or V_f(1-).
# It supports current-level amplitudes for scalar, pseudoscalar, vector,
# axial-vector ("pseudo-vector"), V-A, V+A, and tensor currents; standard
# form-factor projections; helicity amplitudes; and simple leptonic,
# semileptonic, and factorized nonleptonic widths.

HBAR_GEV_S = 6.582119569e-25  # hbar in GeV*s
GF_DEFAULT = 1.1663787e-5      # Fermi constant in GeV^-2


def _canonical_current(current: str) -> str:
    c = str(current).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "scalar": "scalar", "s": "scalar", "scalar_current": "scalar", "one": "scalar", "1": "scalar",
        "pseudoscalar": "pseudoscalar", "p": "pseudoscalar", "ps": "pseudoscalar", "pseudo_scalar": "pseudoscalar",
        "pseudoscalar_current": "pseudoscalar", "gamma5": "pseudoscalar",
        "i_pseudoscalar": "i_pseudoscalar", "ip": "i_pseudoscalar", "i_ps": "i_pseudoscalar", "i_gamma5": "i_pseudoscalar",
        "vector": "vector", "v": "vector", "vector_current": "vector", "gamma_mu": "vector",
        "axial": "axial", "a": "axial", "axial_vector": "axial", "pseudo_vector": "axial",
        "pseudovector": "axial", "axial_current": "axial", "gamma_mu_gamma5": "axial",
        "v_a": "v_minus_a", "vminus_a": "v_minus_a", "left": "v_minus_a",
        "left_chiral": "v_minus_a", "v_minus_a": "v_minus_a", "v-a": "v_minus_a",
        "v_plus_a": "v_plus_a", "v+a": "v_plus_a", "right": "v_plus_a",
        "right_chiral": "v_plus_a",
        "tensor": "tensor", "t": "tensor", "sigma_q": "tensor",
        "tensor_plus": "tensor_plus", "tensor_right": "tensor_plus", "t_plus": "tensor_plus",
        "tensor_minus": "tensor_minus", "tensor_left": "tensor_minus", "t_minus": "tensor_minus",
    }
    if c not in aliases:
        raise ValueError(
            f"Unknown current '{current}'. Use scalar, pseudoscalar, vector, axial, "
            "v_minus_a, v_plus_a, tensor, tensor_plus, or tensor_minus."
        )
    return aliases[c]


def _current_rank(current: str) -> int:
    c = _canonical_current(current)
    if c in ("scalar", "pseudoscalar", "i_pseudoscalar"):
        return 0
    return 1


def _current_matrix(current: str, q, mu: int | None = None, tensor_i_prefactor: bool = False):
    """Return the Dirac matrix for a current. Rank-1 currents require mu."""
    c = _canonical_current(current)
    if c == "scalar":
        return I4
    if c == "pseudoscalar":
        return GAMMA5
    if c == "i_pseudoscalar":
        return 1j * GAMMA5

    if mu is None:
        raise ValueError(f"Current '{current}' needs a Lorentz index mu.")

    if c == "vector":
        return GAMMA[mu]
    if c == "axial":
        return GAMMA[mu] @ GAMMA5
    if c == "v_minus_a":
        return GAMMA[mu] @ (I4 - GAMMA5)
    if c == "v_plus_a":
        return GAMMA[mu] @ (I4 + GAMMA5)
    if c == "tensor":
        return sigma_q(mu, q)
    if c == "tensor_plus":
        M = sigma_q(mu, q) @ (I4 + GAMMA5)
        return 1j * M if tensor_i_prefactor else M
    if c == "tensor_minus":
        M = sigma_q(mu, q) @ (I4 - GAMMA5)
        return 1j * M if tensor_i_prefactor else M
    raise AssertionError("unreachable")


def compute_current_amplitude(
    cfg: CCQMConfig,
    q2: float,
    final_kind: str,
    current: str,
    *,
    tensor_i_prefactor: bool = False,
):
    """
    Compute a current-level CCQM transition amplitude.

    Scope: P_i(0-) -> P_f(0-) or V_f(1-).

    Returned shapes:
      final_kind='P':
        scalar/pseudoscalar -> scalar complex amplitude
        vector/axial/tensor -> M^mu, shape (4,)
      final_kind='V':
        scalar/pseudoscalar -> M^nu, shape (4,), where nu is final-vector index
        vector/axial/tensor -> M^{mu nu}, shape (4,4)

    This function is useful for BSM operators or for checking Ward identities.
    Standard physical form factors should normally be obtained from
    compute_transition_form_factors_general(...).
    """
    fk = str(final_kind).upper().strip()
    if fk in ("P", "PS", "PSEUDOSCALAR"):
        fk = "P"
    elif fk in ("V", "VECTOR"):
        fk = "V"
    else:
        raise ValueError("final_kind must be 'P' or 'V'.")

    rank = _current_rank(current)
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    offsets = (p1, p2, np.zeros(4))
    masses = (cfg.m1, cfg.m2, cfg.m3)

    if fk == "P" and rank == 0:
        def evaluator(p1, p2, q, K, C):
            cur = _current_matrix(current, q, tensor_i_prefactor=tensor_i_prefactor)
            seq = [cur, "N1", GAMMA5, "N3", GAMMA5, "N2"]
            return np.array(trace_average(seq, masses, offsets, K, C))
        return integrate_amplitude(cfg, q2, evaluator, ())

    if fk == "P" and rank == 1:
        def evaluator(p1, p2, q, K, C):
            out = np.zeros(4, dtype=complex)
            for mu in range(4):
                cur = _current_matrix(current, q, mu, tensor_i_prefactor=tensor_i_prefactor)
                seq = [cur, "N1", GAMMA5, "N3", GAMMA5, "N2"]
                out[mu] = trace_average(seq, masses, offsets, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4,))

    if fk == "V" and rank == 0:
        def evaluator(p1, p2, q, K, C):
            out = np.zeros(4, dtype=complex)
            cur = _current_matrix(current, q, tensor_i_prefactor=tensor_i_prefactor)
            for nu in range(4):
                seq = [cur, "N1", GAMMA5, "N3", GAMMA[nu], "N2"]
                out[nu] = trace_average(seq, masses, offsets, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4,))

    if fk == "V" and rank == 1:
        def evaluator(p1, p2, q, K, C):
            out = np.zeros((4, 4), dtype=complex)
            for mu in range(4):
                cur = _current_matrix(current, q, mu, tensor_i_prefactor=tensor_i_prefactor)
                for nu in range(4):
                    seq = [cur, "N1", GAMMA5, "N3", GAMMA[nu], "N2"]
                    out[mu, nu] = trace_average(seq, masses, offsets, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4, 4))

    raise AssertionError("unreachable")


def compute_transition_form_factors_general(
    cfg: CCQMConfig,
    final_kind: str,
    q2: float,
    *,
    include_raw_currents: bool = False,
    currents: list[str] | None = None,
    pv_tensor_i_prefactor: bool = True,
    va_current_factor: float = 1.0,
):
    """
    General P -> P/V form-factor API.

    P -> P standard projections:
      vector current      -> F_plus, F_minus
      scalar current      -> F_S
      tensor sigma.q      -> F_T
      axial/pseudoscalar  -> raw amplitudes, usually parity-suppressed/zero

    P -> V standard projections:
      V-A current         -> A0, A_plus, A_minus, V
      tensor_plus current -> a0, a_plus, g
      vector/axial/scalar/pseudoscalar raw amplitudes are optional.
    """
    fk = str(final_kind).upper().strip()
    if fk in ("P", "PS", "PSEUDOSCALAR"):
        fk = "P"
    elif fk in ("V", "VECTOR"):
        fk = "V"
    else:
        raise ValueError("final_kind must be 'P' or 'V'.")

    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    out = {"final_kind": fk, "q2": q2, "q2_max": (cfg.Mi - cfg.Mf) ** 2}

    if fk == "P":
        M_vector = compute_current_amplitude(cfg, q2, "P", "vector")
        M_scalar = compute_current_amplitude(cfg, q2, "P", "scalar")
        M_tensor = compute_current_amplitude(cfg, q2, "P", "tensor")
        F_plus, F_minus = project_pp_vector(M_vector, p1, p2)
        out.update({
            "F_plus": F_plus,
            "F_minus": F_minus,
            "F_S": M_scalar / (cfg.Mi + cfg.Mf),
            "F_T": project_pp_tensor(M_tensor, p1, p2, cfg.Mi, cfg.Mf),
        })
        # Optional BSM/current-level pieces.
        if include_raw_currents:
            current_list = currents or ["scalar", "pseudoscalar", "vector", "axial", "v_minus_a", "v_plus_a", "tensor"]
            out["raw_currents"] = {
                _canonical_current(c): compute_current_amplitude(cfg, q2, "P", c, tensor_i_prefactor=pv_tensor_i_prefactor)
                for c in current_list
            }
        return out

    # final vector
    M_VA = va_current_factor * compute_current_amplitude(cfg, q2, "V", "v_minus_a")
    M_TP = compute_current_amplitude(cfg, q2, "V", "tensor_plus", tensor_i_prefactor=pv_tensor_i_prefactor)
    A0, A_plus, A_minus, Vff = solve_projection(M_VA, basis_pv_va(p1, p2, cfg.Mi, cfg.Mf), p2, cfg.Mf)
    a0, a_plus, gff = solve_projection(M_TP, basis_pv_tensor(p1, p2), p2, cfg.Mf)
    out.update({
        "A0": A0,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "V": Vff,
        "a0": a0,
        "a_plus": a_plus,
        "g": gff,
    })
    # Common BSW-style conversion.  Useful for comparison with many articles.
    if abs(cfg.Mi - cfg.Mf) > 0:
        out["A1_BSW"] = (cfg.Mi - cfg.Mf) / (cfg.Mi + cfg.Mf) * A0
    out["A2_BSW"] = A_plus
    out["V_BSW"] = Vff

    if include_raw_currents:
        current_list = currents or ["scalar", "pseudoscalar", "vector", "axial", "v_minus_a", "v_plus_a", "tensor_plus", "tensor_minus"]
        out["raw_currents"] = {
            _canonical_current(c): compute_current_amplitude(cfg, q2, "V", c, tensor_i_prefactor=pv_tensor_i_prefactor)
            for c in current_list
        }
    return out


def kallen(x: float, y: float, z: float) -> float:
    """Kallen lambda function lambda(x,y,z)."""
    return x * x + y * y + z * z - 2.0 * (x * y + x * z + y * z)


def momentum_two_body(Mi: float, M1: float, M2: float) -> float:
    """Two-body momentum in rest frame of particle with mass Mi."""
    lam = kallen(Mi * Mi, M1 * M1, M2 * M2)
    return np.sqrt(max(lam, 0.0)) / (2.0 * Mi)


def momentum_q2(Mi: float, Mf: float, q2: float) -> float:
    """Final hadron momentum for P_i -> H_f + virtual W with invariant q2."""
    lam = kallen(Mi * Mi, Mf * Mf, q2)
    return np.sqrt(max(lam, 0.0)) / (2.0 * Mi)


def helicity_amplitudes_pp(ff: dict, Mi: float, Mf: float, q2: float):
    """SM V-A helicity amplitudes for P -> P from F_plus,F_minus."""
    if q2 <= 0:
        raise ValueError("Helicity amplitudes require q2 > 0.")
    p = momentum_q2(Mi, Mf, q2)
    sq = np.sqrt(q2)
    Fp = ff["F_plus"]
    Fm = ff["F_minus"]
    Ht = ((Mi * Mi - Mf * Mf) * Fp + q2 * Fm) / sq
    H0 = 2.0 * Mi * p * Fp / sq
    return {"Ht": Ht, "H0": H0, "Hplus": 0.0, "Hminus": 0.0}


def helicity_amplitudes_pv(ff: dict, Mi: float, Mf: float, q2: float):
    """SM V-A helicity amplitudes for P -> V from A0,A_plus,A_minus,V."""
    if q2 <= 0:
        raise ValueError("Helicity amplitudes require q2 > 0.")
    p = momentum_q2(Mi, Mf, q2)
    sq = np.sqrt(q2)
    D = Mi * Mi - Mf * Mf
    denom = Mi + Mf
    A0 = ff["A0"]
    Ap = ff["A_plus"]
    Am = ff["A_minus"]
    Vff = ff["V"]
    Ht = (Mi * p / (Mf * sq)) * (D * (Ap - A0) + q2 * Am) / denom
    Hplus = (-D * A0 + 2.0 * Mi * p * Vff) / denom
    Hminus = (-D * A0 - 2.0 * Mi * p * Vff) / denom
    H0 = (-D * (D - q2) * A0 + 4.0 * Mi * Mi * p * p * Ap) / (denom * 2.0 * Mf * sq)
    return {"Ht": Ht, "H0": H0, "Hplus": Hplus, "Hminus": Hminus}


def helicity_amplitudes(ff: dict, final_kind: str, Mi: float, Mf: float, q2: float):
    fk = str(final_kind).upper().strip()
    if fk.startswith("P"):
        return helicity_amplitudes_pp(ff, Mi, Mf, q2)
    if fk.startswith("V"):
        return helicity_amplitudes_pv(ff, Mi, Mf, q2)
    raise ValueError("final_kind must be 'P' or 'V'.")


def helicity_structure_functions(H: dict):
    """Return simple structure functions U,L,S,P used in rates."""
    Hp = H.get("Hplus", 0.0)
    Hm = H.get("Hminus", 0.0)
    H0 = H.get("H0", 0.0)
    Ht = H.get("Ht", 0.0)
    return {
        "U": np.real(Hp * np.conjugate(Hp) + Hm * np.conjugate(Hm)),
        "L": np.real(H0 * np.conjugate(H0)),
        "S": np.real(Ht * np.conjugate(Ht)),
        "P": np.real(Hp * np.conjugate(Hp) - Hm * np.conjugate(Hm)),
    }


def leptonic_width(kind: str, M: float, f: float, Vckm: float, ml: float, GF: float = GF_DEFAULT):
    """
    Leptonic weak decay width in GeV.

    kind='P': Gamma(P -> l nu)
    kind='V': Gamma(V -> l nu)
    """
    k = str(kind).upper().strip()
    if ml >= M:
        return 0.0
    if k.startswith("P"):
        return GF * GF / (8.0 * np.pi) * abs(Vckm) ** 2 * f * f * M * ml * ml * (1.0 - ml * ml / (M * M)) ** 2
    if k.startswith("V"):
        return GF * GF / (4.0 * np.pi) * abs(Vckm) ** 2 * f * f * M ** 3 * (1.0 - ml * ml / (M * M)) ** 2 * (1.0 + ml * ml / (2.0 * M * M))
    raise ValueError("kind must be P or V.")


def semileptonic_dgamma_dq2(
    ff: dict,
    final_kind: str,
    Mi: float,
    Mf: float,
    q2: float,
    Vckm: float,
    ml: float = 0.0,
    GF: float = GF_DEFAULT,
):
    """Differential SM semileptonic width dGamma/dq2 for P -> P/V l nu."""
    if q2 <= ml * ml:
        return 0.0
    H = helicity_amplitudes(ff, final_kind, Mi, Mf, q2)
    p = momentum_q2(Mi, Mf, q2)
    hsum = abs(H.get("H0", 0.0)) ** 2 + abs(H.get("Hplus", 0.0)) ** 2 + abs(H.get("Hminus", 0.0)) ** 2
    ht = abs(H.get("Ht", 0.0)) ** 2
    pref = GF * GF * abs(Vckm) ** 2 / ((2.0 * np.pi) ** 3)
    pref *= ((q2 - ml * ml) ** 2 * p) / (12.0 * Mi * Mi * q2)
    val = pref * ((1.0 + ml * ml / (2.0 * q2)) * hsum + (3.0 * ml * ml / (2.0 * q2)) * ht)
    return float(np.real_if_close(val))


def semileptonic_width(
    cfg: CCQMConfig,
    final_kind: str,
    Vckm: float,
    ml: float = 0.0,
    GF: float = GF_DEFAULT,
    n_q2: int = 32,
    q2_min: float | None = None,
    q2_max: float | None = None,
):
    """
    Numerically integrate the SM semileptonic width for P -> P/V l nu.

    This recomputes form factors at Gauss-Legendre q2 nodes.  It is accurate
    but can be slow for large n_q2 and high cfg.n_quad.
    """
    q2lo = ml * ml if q2_min is None else q2_min
    q2hi = (cfg.Mi - cfg.Mf) ** 2 if q2_max is None else q2_max
    if q2hi <= q2lo:
        return {"width_GeV": 0.0, "q2_min": q2lo, "q2_max": q2hi}
    # Avoid exactly q2=0 because helicity amplitudes contain 1/sqrt(q2).
    q2lo_eff = max(q2lo, 1e-9)
    xs, ws = gauss_nodes(q2lo_eff, q2hi, n_q2)
    total = 0.0
    samples = []
    for q2, w in zip(xs, ws):
        ff = compute_transition_form_factors_safe(cfg, final_kind, float(q2), compute_tensor=False)
        dg = semileptonic_dgamma_dq2(ff, final_kind, cfg.Mi, cfg.Mf, float(q2), Vckm, ml, GF)
        total += w * dg
        samples.append({"q2": float(q2), "dGamma_dq2_GeV-1": dg})
    return {"width_GeV": float(total), "q2_min": q2lo, "q2_max": q2hi, "samples": samples}


def nonleptonic_factorized_width(
    H: dict,
    Mi: float,
    Mf: float,
    emitted_kind: str,
    emitted_mass: float,
    emitted_decay_constant: float,
    CKM_product: complex,
    a_eff: complex = 1.0,
    GF: float = GF_DEFAULT,
):
    """
    Two-body nonleptonic width in naive factorization.

    It uses helicity amplitudes for the transition P_i -> H_f evaluated at
    q2 = emitted_mass^2.

    emitted_kind='P': Gamma ∝ |f_P m_P H_t|^2
    emitted_kind='V': Gamma ∝ |f_V m_V|^2 (|H_0|^2+|H_+|^2+|H_-|^2)
    """
    p = momentum_two_body(Mi, Mf, emitted_mass)
    pref = GF * GF / (16.0 * np.pi) * p / (Mi * Mi)
    weak = abs(CKM_product * a_eff * emitted_decay_constant * emitted_mass) ** 2
    ek = str(emitted_kind).upper().strip()
    if ek.startswith("P"):
        hpart = abs(H.get("Ht", 0.0)) ** 2
    elif ek.startswith("V"):
        hpart = abs(H.get("H0", 0.0)) ** 2 + abs(H.get("Hplus", 0.0)) ** 2 + abs(H.get("Hminus", 0.0)) ** 2
    else:
        raise ValueError("emitted_kind must be 'P' or 'V'.")
    return float(np.real_if_close(pref * weak * hpart))


def branching_fraction(width_GeV: float, lifetime_s: float):
    """Convert width in GeV to branching fraction using lifetime in seconds."""
    return float(width_GeV * lifetime_s / HBAR_GEV_S)


def _jsonable(x):
    """Convert numpy/complex objects to JSON-friendly Python objects.

    CCQM loop/projection results are complex-valued internally.  In normal
    physical cases the imaginary part is only roundoff noise, but some raw
    diagnostic amplitudes can be genuinely complex.  JSON cannot store complex
    numbers, so we convert as follows:
      - real-like complex numbers -> float(real)
      - genuinely complex numbers -> {"re": ..., "im": ...}
    """
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        # Preserve genuinely complex entries; do not call real_if_close on the
        # whole array only, because mixed arrays can still contain complex scalars.
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, complex):
        re = float(np.real(x))
        im = float(np.imag(x))
        scale = max(1.0, abs(re), abs(im))
        if abs(im) <= 1e-10 * scale:
            return re
        return {"re": re, "im": im}
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    # Last-resort conversion for objects like Decimal or numpy scalars that were
    # not caught above.  If it still cannot be converted, return a string instead
    # of crashing the app.
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def compute_user_processes(input_data: dict):
    """
    User-facing process engine.

    Input sections:
      global: {lambda_ir, n_quad, Nc, GF}
      mesons: [{name, kind, M, m1, m2, Lambda, optional g}]
      decay_constants: [{name, meson}]
      transitions: [{name, initial, final, final_kind, m1, m2, m3, q2_values, include_raw_currents}]
      leptonic_decays: [{name, meson, lepton_mass, Vckm, lifetime_s?}]
      semileptonic_decays: [{name, transition, lepton_mass, Vckm, n_q2, lifetime_s?}]
      nonleptonic_decays: [{name, transition, emitted, CKM_product, a_eff, lifetime_s?}]

    It returns couplings, decay constants, form factors, helicity amplitudes,
    widths, and optional branching fractions.
    """
    glob = input_data.get("global", {})
    lambda_ir = glob.get("lambda_ir", 0.181)
    n_quad = glob.get("n_quad", 16)
    Nc = glob.get("Nc", 3)
    GF = glob.get("GF", GF_DEFAULT)

    meson_inputs = {m.get("name", f"meson_{i+1}"): dict(m) for i, m in enumerate(input_data.get("mesons", []))}

    # Couplings: use provided g if present, otherwise compute.
    couplings = {}
    for name, item in meson_inputs.items():
        if "g" in item and item["g"] is not None:
            couplings[name] = {"name": name, "kind": _kind_alias(item["kind"]), "g": item["g"], "source": "input"}
        else:
            mc = MesonConfig(
                name=name,
                kind=item["kind"],
                M=item["M"],
                m1=item["m1"],
                m2=item["m2"],
                Lambda=item["Lambda"],
                lambda_ir=item.get("lambda_ir", lambda_ir),
                Nc=item.get("Nc", Nc),
                n_quad=item.get("n_quad", n_quad),
            )
            couplings[name] = compute_coupling_constant(mc, derivative_step=item.get("derivative_step", 1e-4))

    results = {
        "couplings": couplings,
        "decay_constants": {},
        "transitions": {},
        "leptonic_decays": {},
        "semileptonic_decays": {},
        "nonleptonic_decays": {},
        "notes": [
            "Scope: pseudoscalar initial meson to pseudoscalar or vector final meson.",
            "Currents supported at amplitude level: scalar, pseudoscalar, vector, axial/pseudo-vector, V-A, V+A, tensor.",
            "Semileptonic and nonleptonic width formulas use SM V-A helicity amplitudes and naive factorization.",
        ],
    }

    # Decay constants requested explicitly, plus cache helper.
    def get_decay_constant_for(meson_name: str):
        if meson_name in results["decay_constants"]:
            return results["decay_constants"][meson_name]
        src = meson_inputs[meson_name]
        dc = DecayConstantConfig(
            kind=src["kind"], M=src["M"], m1=src["m1"], m2=src["m2"], Lambda=src["Lambda"],
            g=couplings[meson_name]["g"], lambda_ir=src.get("lambda_ir", lambda_ir),
            Nc=src.get("Nc", Nc), n_quad=src.get("n_quad", n_quad),
        )
        results["decay_constants"][meson_name] = compute_decay_constant(dc)
        return results["decay_constants"][meson_name]

    for item in input_data.get("decay_constants", []):
        meson_name = item.get("meson", item.get("name"))
        out_name = item.get("name", meson_name)
        if meson_name in meson_inputs:
            results["decay_constants"][out_name] = get_decay_constant_for(meson_name)
        else:
            dc = DecayConstantConfig(
                kind=item["kind"], M=item["M"], m1=item["m1"], m2=item["m2"], Lambda=item["Lambda"],
                g=item.get("g", 1.0), lambda_ir=item.get("lambda_ir", lambda_ir),
                Nc=item.get("Nc", Nc), n_quad=item.get("n_quad", n_quad),
            )
            results["decay_constants"][out_name] = compute_decay_constant(dc)

    transition_cfgs = {}
    for item in input_data.get("transitions", []):
        name = item.get("name", f"transition_{len(results['transitions'])+1}")
        initial_name = item.get("initial")
        final_name = item.get("final")
        initial = meson_inputs.get(initial_name, {})
        final = meson_inputs.get(final_name, {})
        final_kind = item.get("final_kind", final.get("kind"))
        cfg = CCQMConfig(
            Mi=item.get("Mi", initial.get("M")),
            Mf=item.get("Mf", final.get("M")),
            m1=item["m1"], m2=item["m2"], m3=item["m3"],
            Lambda_i=item.get("Lambda_i", initial.get("Lambda")),
            Lambda_f=item.get("Lambda_f", final.get("Lambda")),
            g_i=item.get("g_i", couplings.get(initial_name, {}).get("g", 1.0)),
            g_f=item.get("g_f", couplings.get(final_name, {}).get("g", 1.0)),
            lambda_ir=item.get("lambda_ir", lambda_ir), Nc=item.get("Nc", Nc), n_quad=item.get("n_quad", n_quad),
        )
        transition_cfgs[name] = (cfg, final_kind, item)
        tres = {
            "initial": initial_name, "final": final_name, "final_kind": final_kind,
            "q2_max": (cfg.Mi - cfg.Mf) ** 2, "g_i": cfg.g_i, "g_f": cfg.g_f,
            "form_factors": {}, "helicity": {},
        }
        for q2 in item.get("q2_values", [0.0]):
            q2f = float(q2)
            ff = compute_transition_form_factors_safe(
                cfg, final_kind, q2f,
                include_raw_currents=item.get("include_raw_currents", False),
                currents=item.get("currents"),
                pv_tensor_i_prefactor=item.get("pv_tensor_i_prefactor", True),
                va_current_factor=item.get("va_current_factor", 1.0),
            )
            tres["form_factors"][str(q2)] = ff
            if q2f > 0:
                try:
                    tres["helicity"][str(q2)] = helicity_amplitudes(ff, final_kind, cfg.Mi, cfg.Mf, q2f)
                except Exception as exc:
                    tres["helicity"][str(q2)] = {"error": str(exc)}
        results["transitions"][name] = tres

    for item in input_data.get("leptonic_decays", []):
        name = item.get("name", f"leptonic_{len(results['leptonic_decays'])+1}")
        meson_name = item["meson"]
        mes = meson_inputs[meson_name]
        dc = get_decay_constant_for(meson_name)
        width = leptonic_width(mes["kind"], mes["M"], dc["f_GeV"], item["Vckm"], item.get("lepton_mass", 0.0), GF)
        row = {"meson": meson_name, "width_GeV": width, "f_GeV": dc["f_GeV"]}
        if "lifetime_s" in item:
            row["branching_fraction"] = branching_fraction(width, item["lifetime_s"])
        results["leptonic_decays"][name] = row

    for item in input_data.get("semileptonic_decays", []):
        name = item.get("name", f"semileptonic_{len(results['semileptonic_decays'])+1}")
        tname = item["transition"]
        cfg, final_kind, _ = transition_cfgs[tname]
        row = semileptonic_width(
            cfg, final_kind, item["Vckm"], ml=item.get("lepton_mass", 0.0), GF=GF,
            n_q2=item.get("n_q2", 24), q2_min=item.get("q2_min"), q2_max=item.get("q2_max"),
        )
        if "lifetime_s" in item:
            row["branching_fraction"] = branching_fraction(row["width_GeV"], item["lifetime_s"])
        results["semileptonic_decays"][name] = row

    for item in input_data.get("nonleptonic_decays", []):
        name = item.get("name", f"nonleptonic_{len(results['nonleptonic_decays'])+1}")
        tname = item["transition"]
        emitted = item["emitted"]
        cfg, final_kind, _ = transition_cfgs[tname]
        em = meson_inputs[emitted]
        dc = get_decay_constant_for(emitted)
        q2 = em["M"] * em["M"]
        ff = compute_transition_form_factors_safe(cfg, final_kind, q2, compute_tensor=False)
        H = helicity_amplitudes(ff, final_kind, cfg.Mi, cfg.Mf, q2)
        width = nonleptonic_factorized_width(
            H, cfg.Mi, cfg.Mf, em["kind"], em["M"], dc["f_GeV"],
            item["CKM_product"], item.get("a_eff", 1.0), GF=GF,
        )
        row = {"transition": tname, "emitted": emitted, "q2": q2, "width_GeV": width, "helicity": H}
        if "lifetime_s" in item:
            row["branching_fraction"] = branching_fraction(width, item["lifetime_s"])
        results["nonleptonic_decays"][name] = row

    return _jsonable(results)

# ---------------------------------------------------------------------
# Checked/safe convenience API added after 3-process benchmark
# ---------------------------------------------------------------------

def compute_pv_va_form_factors_only(cfg: CCQMConfig, q2: float, va_current_factor: float = 1.0):
    """
    Compute only the P->V V-A form factors A0,A_plus,A_minus,V.

    This is useful at q2=0 because the tensor basis for a0,a_plus,g contains
    explicit 1/q2 factors, while the V-A basis is regular at q2=0.
    It also avoids the expensive tensor-current loop when only A1,A2,V are
    needed for article comparisons or nonleptonic P->V emissions.
    """
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    M_VA = va_current_factor * compute_current_amplitude(cfg, q2, "V", "v_minus_a")
    A0, A_plus, A_minus, Vff = solve_projection(
        M_VA, basis_pv_va(p1, p2, cfg.Mi, cfg.Mf), p2, cfg.Mf
    )
    out = {
        "final_kind": "V",
        "q2": q2,
        "q2_max": (cfg.Mi - cfg.Mf) ** 2,
        "A0": A0,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "V": Vff,
        "A1_BSW": (cfg.Mi - cfg.Mf) / (cfg.Mi + cfg.Mf) * A0 if abs(cfg.Mi + cfg.Mf) > 0 else np.nan,
        "A2_BSW": A_plus,
        "V_BSW": Vff,
        "note": "V-A form factors only; tensor form factors are not computed by this helper.",
    }
    return out


def compute_pv_tensor_form_factors_only(
    cfg: CCQMConfig,
    q2: float,
    pv_tensor_i_prefactor: bool = True,
):
    """
    Compute only the P->V tensor form factors a0,a_plus,g.

    q2 must be positive because the standard tensor basis has explicit 1/q2.
    For q2=0, evaluate at several small positive q2 values and extrapolate.
    """
    if abs(q2) < 1e-12:
        raise ValueError("P->V tensor form factors need q2>0; use a small-q2 extrapolation.")
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    M_TP = compute_current_amplitude(cfg, q2, "V", "tensor_plus", tensor_i_prefactor=pv_tensor_i_prefactor)
    a0, a_plus, gff = solve_projection(M_TP, basis_pv_tensor(p1, p2), p2, cfg.Mf)
    return {
        "final_kind": "V",
        "q2": q2,
        "q2_max": (cfg.Mi - cfg.Mf) ** 2,
        "a0": a0,
        "a_plus": a_plus,
        "g": gff,
        "T2_BSW": a0,
        "T1_BSW": gff,
    }


def compute_transition_form_factors_safe(
    cfg: CCQMConfig,
    final_kind: str,
    q2: float,
    *,
    compute_tensor: bool = True,
    include_raw_currents: bool = False,
    currents: list[str] | None = None,
    pv_tensor_i_prefactor: bool = True,
    va_current_factor: float = 1.0,
):
    """
    Safer user-facing wrapper for P->P/P->V form factors.

    Difference from compute_transition_form_factors_general:
      - P->V at q2=0 no longer fails when only V-A form factors are needed.
      - Set compute_tensor=False to skip a0,a_plus,g and speed up P->V runs.
      - If compute_tensor=True at q2=0, tensor form factors are skipped with a note.
    """
    fk = str(final_kind).upper().strip()
    if fk in ("P", "PS", "PSEUDOSCALAR"):
        return compute_transition_form_factors_general(
            cfg,
            "P",
            q2,
            include_raw_currents=include_raw_currents,
            currents=currents,
            pv_tensor_i_prefactor=pv_tensor_i_prefactor,
            va_current_factor=va_current_factor,
        )
    if fk not in ("V", "VECTOR"):
        raise ValueError("final_kind must be 'P' or 'V'.")

    out = compute_pv_va_form_factors_only(cfg, q2, va_current_factor=va_current_factor)

    if compute_tensor:
        if abs(q2) < 1e-12:
            out["tensor_note"] = (
                "Skipped P->V tensor form factors at q2=0 because the standard basis contains 1/q2. "
                "Evaluate at small positive q2 values and extrapolate to q2=0."
            )
        else:
            out.update(compute_pv_tensor_form_factors_only(cfg, q2, pv_tensor_i_prefactor=pv_tensor_i_prefactor))

    if include_raw_currents:
        current_list = currents or ["scalar", "pseudoscalar", "vector", "axial", "v_minus_a", "v_plus_a"]
        if compute_tensor and abs(q2) > 1e-12:
            current_list = current_list + ["tensor_plus", "tensor_minus"]
        out["raw_currents"] = {
            _canonical_current(c): compute_current_amplitude(cfg, q2, "V", c, tensor_i_prefactor=pv_tensor_i_prefactor)
            for c in current_list
        }
    return out


# ---------------------------------------------------------------------
# Validation/diagnostic helpers added for precision user app
# ---------------------------------------------------------------------

VALIDATION_OUTPUT_CATEGORIES = [
    "couplings",
    "decay_constants",
    "transition_form_factors",
    "raw_currents",
    "helicity_amplitudes",
    "helicity_structure_functions",
    "leptonic_widths",
    "semileptonic_widths",
    "nonleptonic_factorized_widths",
]


def classify_relative_difference(value: float, reference: float, pass_pct: float = 5.0, warn_pct: float = 20.0):
    """Small helper for benchmark reports."""
    if reference == 0 or reference is None:
        return "info", None
    diff = 100.0 * (value - reference) / reference
    ad = abs(diff)
    if ad <= pass_pct:
        return "pass", diff
    if ad <= warn_pct:
        return "warn", diff
    return "fail", diff


def required_a_eff_for_branching_fraction(width_at_a_eff_1: float, target_branching_fraction: float, lifetime_s: float):
    """Return the effective factor a_eff needed to reproduce a target branching fraction."""
    bf1 = branching_fraction(width_at_a_eff_1, lifetime_s)
    if bf1 <= 0:
        return np.nan
    return np.sqrt(target_branching_fraction / bf1)

# ---------------------------------------------------------------------
# Fast trace-term backend: avoids recomputing Dirac traces at every quadrature node.
# This overrides compute_current_amplitude with an algebraically equivalent version.
# ---------------------------------------------------------------------

def _trace_terms_for_sequence(sequence, masses):
    """Precompute Dirac trace coefficients for a sequence containing N1/N2/N3 labels."""
    mass = {"N1": masses[0], "N2": masses[1], "N3": masses[2]}
    terms = []
    def rec(i, M, selected):
        if i == len(sequence):
            coeff = np.trace(M)
            if abs(coeff) > 1e-14:
                terms.append((coeff, tuple(selected)))
            return
        item = sequence[i]
        if isinstance(item, str):
            rec(i + 1, M @ (mass[item] * I4), selected)
            for lam in range(4):
                rec(i + 1, M @ GAMMA[lam], selected + [(item, lam)])
        else:
            rec(i + 1, M @ item, selected)
    rec(0, I4, [])
    return terms


def _eval_trace_terms(terms, offset_map, K, C):
    total = 0.0 + 0.0j
    for coeff, selected_labels in terms:
        selected = [(offset_map[label], lam) for (label, lam) in selected_labels]
        total += coeff * moment_lower(selected, K, C)
    return total


def compute_current_amplitude(
    cfg: CCQMConfig,
    q2: float,
    final_kind: str,
    current: str,
    *,
    tensor_i_prefactor: bool = False,
):
    """
    Fast current-level CCQM transition amplitude.

    This is algebraically equivalent to the original recursive implementation,
    but Dirac trace coefficients are precomputed once per q2/current instead of
    at every Schwinger/simplex integration node.
    """
    fk = str(final_kind).upper().strip()
    if fk in ("P", "PS", "PSEUDOSCALAR"):
        fk = "P"
    elif fk in ("V", "VECTOR"):
        fk = "V"
    else:
        raise ValueError("final_kind must be 'P' or 'V'.")

    rank = _current_rank(current)
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    offsets = (p1, p2, np.zeros(4))
    offset_map = {"N1": offsets[0], "N2": offsets[1], "N3": offsets[2]}
    masses = (cfg.m1, cfg.m2, cfg.m3)
    q = p1 - p2

    if fk == "P" and rank == 0:
        cur = _current_matrix(current, q, tensor_i_prefactor=tensor_i_prefactor)
        terms = _trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA5, "N2"], masses)
        def evaluator(p1_, p2_, q_, K, C):
            return np.array(_eval_trace_terms(terms, offset_map, K, C))
        return integrate_amplitude(cfg, q2, evaluator, ())

    if fk == "P" and rank == 1:
        all_terms = []
        for mu in range(4):
            cur = _current_matrix(current, q, mu, tensor_i_prefactor=tensor_i_prefactor)
            all_terms.append(_trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA5, "N2"], masses))
        def evaluator(p1_, p2_, q_, K, C):
            out = np.zeros(4, dtype=complex)
            for mu in range(4):
                out[mu] = _eval_trace_terms(all_terms[mu], offset_map, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4,))

    if fk == "V" and rank == 0:
        cur = _current_matrix(current, q, tensor_i_prefactor=tensor_i_prefactor)
        all_terms = []
        for nu in range(4):
            all_terms.append(_trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA[nu], "N2"], masses))
        def evaluator(p1_, p2_, q_, K, C):
            out = np.zeros(4, dtype=complex)
            for nu in range(4):
                out[nu] = _eval_trace_terms(all_terms[nu], offset_map, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4,))

    if fk == "V" and rank == 1:
        all_terms = [[None for _ in range(4)] for __ in range(4)]
        for mu in range(4):
            cur = _current_matrix(current, q, mu, tensor_i_prefactor=tensor_i_prefactor)
            for nu in range(4):
                all_terms[mu][nu] = _trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA[nu], "N2"], masses)
        def evaluator(p1_, p2_, q_, K, C):
            out = np.zeros((4, 4), dtype=complex)
            for mu in range(4):
                for nu in range(4):
                    out[mu, nu] = _eval_trace_terms(all_terms[mu][nu], offset_map, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4, 4))

    raise AssertionError("unreachable")


# =====================================================================
# Current-resolved form-factor registry and projections
# =====================================================================
# This layer distinguishes two concepts:
#   (1) raw current amplitudes, e.g. M^mu or M^{mu nu};
#   (2) scalar invariant form factors obtained by projecting the amplitude
#       onto allowed Lorentz structures.
# It is intended for user-facing apps where users choose a current type.

CURRENT_FORM_FACTOR_SCHEMA = {
    "P_to_P": {
        "scalar": {
            "operator": "1",
            "projected_form_factors": ["F_S"],
            "definition": "<P_f|S|P_i> = (M_i + M_f) F_S(q2)",
            "standard": True,
        },
        "pseudoscalar": {
            "operator": "gamma5 or i gamma5",
            "projected_form_factors": ["F_P5_raw", "F_P5_norm"],
            "definition": "No independent standard P->P pseudoscalar form factor in parity-conserving QCD; raw amplitude is returned for BSM checks.",
            "standard": False,
        },
        "vector": {
            "operator": "gamma^mu",
            "projected_form_factors": ["F_plus", "F_minus"],
            "definition": "<P_f|V^mu|P_i> = P^mu F_plus(q2) + q^mu F_minus(q2)",
            "standard": True,
        },
        "axial": {
            "operator": "gamma^mu gamma5",
            "projected_form_factors": ["A_plus_raw", "A_minus_raw"],
            "definition": "P->P axial current has no independent parity-allowed form factor; projected raw values should be numerically zero.",
            "standard": False,
        },
        "v_minus_a": {
            "operator": "gamma^mu(1-gamma5)",
            "projected_form_factors": ["F_plus", "F_minus", "A_plus_raw", "A_minus_raw"],
            "definition": "Left current decomposed into vector part plus axial parity-check part.",
            "standard": True,
        },
        "v_plus_a": {
            "operator": "gamma^mu(1+gamma5)",
            "projected_form_factors": ["F_plus", "F_minus", "A_plus_raw", "A_minus_raw"],
            "definition": "Right current decomposed into vector part plus axial parity-check part.",
            "standard": False,
        },
        "tensor": {
            "operator": "sigma^{mu nu} q_nu",
            "projected_form_factors": ["F_T"],
            "definition": "<P_f|T^mu|P_i> = i/(M_i+M_f) [q2 P^mu - (P.q) q^mu] F_T(q2)",
            "standard": True,
        },
    },
    "P_to_V": {
        "scalar": {
            "operator": "1",
            "projected_form_factors": ["S_PV_raw", "S_PV"],
            "definition": "<V_f(eps)|S|P_i> = (eps*.P) S_PV(q2); expected zero for pure scalar by vector-current Ward identity in many cases.",
            "standard": False,
        },
        "pseudoscalar": {
            "operator": "gamma5 or i gamma5",
            "projected_form_factors": ["P_PV_raw", "P_PV"],
            "definition": "<V_f(eps)|P5|P_i> = (eps*.P) P_PV(q2); related to the divergence of the axial current by equations of motion.",
            "standard": False,
        },
        "vector": {
            "operator": "gamma^mu",
            "projected_form_factors": ["V"],
            "definition": "Vector-current part of <V_f|gamma^mu|P_i>; only the epsilon-tensor structure V(q2) is parity allowed.",
            "standard": True,
        },
        "axial": {
            "operator": "gamma^mu gamma5",
            "projected_form_factors": ["A0", "A_plus", "A_minus"],
            "definition": "Axial-current part parameterized by A0(q2), A_plus(q2), A_minus(q2) in CCQM convention.",
            "standard": True,
        },
        "v_minus_a": {
            "operator": "gamma^mu(1-gamma5)",
            "projected_form_factors": ["A0", "A_plus", "A_minus", "V"],
            "definition": "Full left-handed V-A current used for SM semileptonic and nonleptonic amplitudes.",
            "standard": True,
        },
        "v_plus_a": {
            "operator": "gamma^mu(1+gamma5)",
            "projected_form_factors": ["A0", "A_plus", "A_minus", "V"],
            "definition": "Right-handed V+A current; signs differ in the axial part depending on convention, raw amplitude is also available.",
            "standard": False,
        },
        "tensor_plus": {
            "operator": "sigma^{mu nu} q_nu (1+gamma5)",
            "projected_form_factors": ["a0", "a_plus", "g"],
            "definition": "Rare-decay tensor basis in CCQM convention; contains 1/q2, so use q2>0 or extrapolate to q2=0.",
            "standard": True,
        },
        "tensor_minus": {
            "operator": "sigma^{mu nu} q_nu (1-gamma5)",
            "projected_form_factors": ["a0", "a_plus", "g"],
            "definition": "Left-chiral tensor current; same Lorentz basis as tensor_plus, different chiral projector.",
            "standard": False,
        },
    },
}


def get_current_form_factor_schema(final_kind: str | None = None):
    """Return the current -> form-factor map used by the user interface."""
    if final_kind is None:
        return CURRENT_FORM_FACTOR_SCHEMA
    fk = str(final_kind).upper().strip()
    key = "P_to_V" if fk.startswith("V") else "P_to_P"
    return CURRENT_FORM_FACTOR_SCHEMA[key]


def _project_pv_spin0_current(Mnu, p1, p2, Mi: float, Mf: float):
    """
    Project a rank-0 current P_i -> V_f amplitude M^nu onto
        eps*_nu M^nu = (eps*.P) F(q2).

    Since eps*.p2=0, only the component transverse to p2 is physical.
    """
    P = p1 + p2
    p2_lower = lower(p2)
    Pi_lower = -G + np.outer(p2_lower, p2_lower) / (Mf * Mf)

    def inner_vec(A, B):
        s = 0.0 + 0.0j
        for nu in range(4):
            for rho in range(4):
                s += A[nu] * Pi_lower[nu, rho] * B[rho]
        return s

    denom = inner_vec(P, P)
    if abs(denom) < 1e-14:
        return np.nan + 0.0j
    return inner_vec(P, Mnu) / denom


def _split_va_solution(sol):
    """Return dict from solve_projection result in CCQM P->V VA basis."""
    return {"A0": sol[0], "A_plus": sol[1], "A_minus": sol[2], "V": sol[3]}


def compute_form_factors_by_current(
    cfg: CCQMConfig,
    final_kind: str,
    q2: float,
    currents: list[str] | None = None,
    *,
    pv_tensor_i_prefactor: bool = True,
    va_current_factor: float = 1.0,
    include_raw_amplitudes: bool = False,
):
    """
    Compute form factors grouped by current type.

    Returns a nested dictionary:
      {
        'transition_type': 'P_to_P' or 'P_to_V',
        'q2': ...,
        'by_current': {
            'scalar': {'form_factors': {...}, 'raw_amplitude': ... optional, 'schema': ...},
            ...
        }
      }

    This is the preferred API for a user-facing calculator where the user
    selects current(s): scalar, pseudoscalar, vector, axial/pseudo-vector,
    V-A, V+A, tensor, tensor_plus, tensor_minus.
    """
    fk0 = str(final_kind).upper().strip()
    fk = "V" if fk0.startswith("V") else "P"
    transition_type = "P_to_V" if fk == "V" else "P_to_P"
    schema = get_current_form_factor_schema(fk)
    if currents is None:
        currents = list(schema.keys())

    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    out = {
        "transition_type": transition_type,
        "final_kind": fk,
        "q2": q2,
        "q2_max": (cfg.Mi - cfg.Mf) ** 2,
        "by_current": {},
    }

    for cur_in in currents:
        cur = _canonical_current(cur_in)
        entry = {
            "operator": schema.get(cur, {}).get("operator", cur),
            "definition": schema.get(cur, {}).get("definition", "User-requested current."),
            "standard": schema.get(cur, {}).get("standard", False),
            "form_factors": {},
        }

        if fk == "P":
            if cur == "scalar":
                M = compute_current_amplitude(cfg, q2, "P", "scalar")
                entry["form_factors"] = {"F_S": M / (cfg.Mi + cfg.Mf)}
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur in ("pseudoscalar", "i_pseudoscalar"):
                M = compute_current_amplitude(cfg, q2, "P", cur)
                entry["form_factors"] = {
                    "F_P5_raw": M,
                    "F_P5_norm": M / (cfg.Mi + cfg.Mf),
                    "parity_expected_zero": True,
                }
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur == "vector":
                M = compute_current_amplitude(cfg, q2, "P", "vector")
                Fp, Fm = project_pp_vector(M, p1, p2)
                entry["form_factors"] = {"F_plus": Fp, "F_minus": Fm}
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur == "axial":
                M = compute_current_amplitude(cfg, q2, "P", "axial")
                Ap, Am = project_pp_vector(M, p1, p2)
                entry["form_factors"] = {
                    "A_plus_raw": Ap,
                    "A_minus_raw": Am,
                    "parity_expected_zero": True,
                }
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur in ("v_minus_a", "v_plus_a"):
                M = compute_current_amplitude(cfg, q2, "P", cur)
                Xp, Xm = project_pp_vector(M, p1, p2)
                # Also provide separated pieces for clarity.
                Mv = compute_current_amplitude(cfg, q2, "P", "vector")
                Ma = compute_current_amplitude(cfg, q2, "P", "axial")
                Fp, Fm = project_pp_vector(Mv, p1, p2)
                Ap, Am = project_pp_vector(Ma, p1, p2)
                entry["form_factors"] = {
                    "combined_plus": Xp,
                    "combined_minus": Xm,
                    "F_plus_vector_part": Fp,
                    "F_minus_vector_part": Fm,
                    "A_plus_raw": Ap,
                    "A_minus_raw": Am,
                }
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur in ("tensor", "tensor_plus", "tensor_minus"):
                tensor_current = "tensor" if cur == "tensor" else cur
                M = compute_current_amplitude(cfg, q2, "P", tensor_current, tensor_i_prefactor=pv_tensor_i_prefactor)
                entry["form_factors"] = {"F_T": project_pp_tensor(M, p1, p2, cfg.Mi, cfg.Mf)}
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M
            else:
                entry["warning"] = f"Current {cur} is not supported for P_to_P projection."

        else:  # P -> V
            if cur == "scalar":
                M = compute_current_amplitude(cfg, q2, "V", "scalar")
                entry["form_factors"] = {"S_PV": _project_pv_spin0_current(M, p1, p2, cfg.Mi, cfg.Mf), "S_PV_raw_vector": M}
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur in ("pseudoscalar", "i_pseudoscalar"):
                M = compute_current_amplitude(cfg, q2, "V", cur)
                entry["form_factors"] = {"P_PV": _project_pv_spin0_current(M, p1, p2, cfg.Mi, cfg.Mf), "P_PV_raw_vector": M}
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur == "vector":
                M = compute_current_amplitude(cfg, q2, "V", "vector")
                sol = solve_projection(M, basis_pv_va(p1, p2, cfg.Mi, cfg.Mf), p2, cfg.Mf)
                entry["form_factors"] = {"V": sol[3], "A0_raw_zero": sol[0], "A_plus_raw_zero": sol[1], "A_minus_raw_zero": sol[2]}
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur == "axial":
                M = compute_current_amplitude(cfg, q2, "V", "axial")
                sol = solve_projection(M, basis_pv_va(p1, p2, cfg.Mi, cfg.Mf), p2, cfg.Mf)
                entry["form_factors"] = {"A0": sol[0], "A_plus": sol[1], "A_minus": sol[2], "V_raw_zero": sol[3]}
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur in ("v_minus_a", "v_plus_a"):
                M = va_current_factor * compute_current_amplitude(cfg, q2, "V", cur)
                sol = solve_projection(M, basis_pv_va(p1, p2, cfg.Mi, cfg.Mf), p2, cfg.Mf)
                entry["form_factors"] = _split_va_solution(sol)
                if abs(cfg.Mi - cfg.Mf) > 0:
                    entry["form_factors"]["A1_BSW"] = (cfg.Mi - cfg.Mf) / (cfg.Mi + cfg.Mf) * sol[0]
                entry["form_factors"]["A2_BSW"] = sol[1]
                entry["form_factors"]["V_BSW"] = sol[3]
                if include_raw_amplitudes:
                    entry["raw_amplitude"] = M

            elif cur in ("tensor", "tensor_plus", "tensor_minus"):
                if abs(q2) < 1e-12:
                    entry["warning"] = "P->V tensor basis contains 1/q2. Compute at q2>0 and extrapolate to q2=0."
                    entry["form_factors"] = {"a0": None, "a_plus": None, "g": None}
                else:
                    tensor_current = "tensor_plus" if cur == "tensor" else cur
                    M = compute_current_amplitude(cfg, q2, "V", tensor_current, tensor_i_prefactor=pv_tensor_i_prefactor)
                    sol = solve_projection(M, basis_pv_tensor(p1, p2), p2, cfg.Mf)
                    entry["form_factors"] = {"a0": sol[0], "a_plus": sol[1], "g": sol[2]}
                    if include_raw_amplitudes:
                        entry["raw_amplitude"] = M
            else:
                entry["warning"] = f"Current {cur} is not supported for P_to_V projection."

        out["by_current"][cur] = entry

    return out


def compute_form_factors_by_current_jsonable(*args, **kwargs):
    """JSON-friendly wrapper for compute_form_factors_by_current."""
    return _jsonable(compute_form_factors_by_current(*args, **kwargs))


# ---------------------------------------------------------------------
# v5 precision speed patch appended from validated v3 backend
# ---------------------------------------------------------------------
# Fast trace-term backend: avoids recomputing Dirac traces at every quadrature node.
# This overrides compute_current_amplitude with an algebraically equivalent version.
# ---------------------------------------------------------------------

def _trace_terms_for_sequence(sequence, masses):
    """Precompute Dirac trace coefficients for a sequence containing N1/N2/N3 labels."""
    mass = {"N1": masses[0], "N2": masses[1], "N3": masses[2]}
    terms = []
    def rec(i, M, selected):
        if i == len(sequence):
            coeff = np.trace(M)
            if abs(coeff) > 1e-14:
                terms.append((coeff, tuple(selected)))
            return
        item = sequence[i]
        if isinstance(item, str):
            rec(i + 1, M @ (mass[item] * I4), selected)
            for lam in range(4):
                rec(i + 1, M @ GAMMA[lam], selected + [(item, lam)])
        else:
            rec(i + 1, M @ item, selected)
    rec(0, I4, [])
    return terms


def _eval_trace_terms(terms, offset_map, K, C):
    total = 0.0 + 0.0j
    for coeff, selected_labels in terms:
        selected = [(offset_map[label], lam) for (label, lam) in selected_labels]
        total += coeff * moment_lower(selected, K, C)
    return total


def compute_current_amplitude(
    cfg: CCQMConfig,
    q2: float,
    final_kind: str,
    current: str,
    *,
    tensor_i_prefactor: bool = False,
):
    """
    Fast current-level CCQM transition amplitude.

    This is algebraically equivalent to the original recursive implementation,
    but Dirac trace coefficients are precomputed once per q2/current instead of
    at every Schwinger/simplex integration node.
    """
    fk = str(final_kind).upper().strip()
    if fk in ("P", "PS", "PSEUDOSCALAR"):
        fk = "P"
    elif fk in ("V", "VECTOR"):
        fk = "V"
    else:
        raise ValueError("final_kind must be 'P' or 'V'.")

    rank = _current_rank(current)
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    offsets = (p1, p2, np.zeros(4))
    offset_map = {"N1": offsets[0], "N2": offsets[1], "N3": offsets[2]}
    masses = (cfg.m1, cfg.m2, cfg.m3)
    q = p1 - p2

    if fk == "P" and rank == 0:
        cur = _current_matrix(current, q, tensor_i_prefactor=tensor_i_prefactor)
        terms = _trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA5, "N2"], masses)
        def evaluator(p1_, p2_, q_, K, C):
            return np.array(_eval_trace_terms(terms, offset_map, K, C))
        return integrate_amplitude(cfg, q2, evaluator, ())

    if fk == "P" and rank == 1:
        all_terms = []
        for mu in range(4):
            cur = _current_matrix(current, q, mu, tensor_i_prefactor=tensor_i_prefactor)
            all_terms.append(_trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA5, "N2"], masses))
        def evaluator(p1_, p2_, q_, K, C):
            out = np.zeros(4, dtype=complex)
            for mu in range(4):
                out[mu] = _eval_trace_terms(all_terms[mu], offset_map, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4,))

    if fk == "V" and rank == 0:
        cur = _current_matrix(current, q, tensor_i_prefactor=tensor_i_prefactor)
        all_terms = []
        for nu in range(4):
            all_terms.append(_trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA[nu], "N2"], masses))
        def evaluator(p1_, p2_, q_, K, C):
            out = np.zeros(4, dtype=complex)
            for nu in range(4):
                out[nu] = _eval_trace_terms(all_terms[nu], offset_map, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4,))

    if fk == "V" and rank == 1:
        all_terms = [[None for _ in range(4)] for __ in range(4)]
        for mu in range(4):
            cur = _current_matrix(current, q, mu, tensor_i_prefactor=tensor_i_prefactor)
            for nu in range(4):
                all_terms[mu][nu] = _trace_terms_for_sequence([cur, "N1", GAMMA5, "N3", GAMMA[nu], "N2"], masses)
        def evaluator(p1_, p2_, q_, K, C):
            out = np.zeros((4, 4), dtype=complex)
            for mu in range(4):
                for nu in range(4):
                    out[mu, nu] = _eval_trace_terms(all_terms[mu][nu], offset_map, K, C)
            return out
        return integrate_amplitude(cfg, q2, evaluator, (4, 4))

    raise AssertionError("unreachable")

# ---------------------------------------------------------------------
# v6 q2-grid helper API for user interfaces
# ---------------------------------------------------------------------

def physical_q2_max(Mi: float, Mf: float) -> float:
    """Physical upper endpoint q2_max=(Mi-Mf)^2 for P_i -> H_f transitions."""
    return float((Mi - Mf) ** 2)


def _q2_parse_token(token, q2_max: float | None = None) -> float:
    """Parse a q2 token. Supports numeric strings and q2max/physical/max aliases."""
    if isinstance(token, (int, float, np.floating)):
        return float(token)
    s = str(token).strip().lower()
    if s in ("q2max", "qmax", "max", "physical", "physical_max"):
        if q2_max is None:
            raise ValueError("q2max token requires Mi and Mf or q2_max.")
        return float(q2_max)
    return float(s)


def parse_q2_values(value, *, q2_max: float | None = None) -> list[float]:
    """
    Parse a discrete q2 list.

    Accepted examples:
        [0, 1, 2]
        "0, 1, 2"
        "0 1 2"
        "0; 1; q2max"
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_q2_parse_token(x, q2_max=q2_max) for x in value]
    s = str(value).strip()
    if not s:
        return []
    for sep in [";", "\n", "\t"]:
        s = s.replace(sep, ",")
    # If user typed space-separated values but no commas, treat spaces as separators.
    if "," not in s:
        parts = [p for p in s.split(" ") if p.strip()]
    else:
        parts = [p for p in s.split(",") if p.strip()]
    return [_q2_parse_token(p, q2_max=q2_max) for p in parts]


def make_q2_grid(
    *,
    Mi: float,
    Mf: float,
    mode: str = "list",
    q2_values=None,
    q2_min=None,
    q2_max=None,
    q2_points: int = 1,
    endpoint: bool = True,
    unique: bool = True,
    validate_physical: bool = True,
    clip_to_physical: bool = False,
) -> list[float]:
    """
    Build q2 values from either a discrete list or an evenly spaced range.

    Parameters
    ----------
    mode:
        "list" / "discrete" / "manual": use q2_values.
        "range" / "grid" / "linspace": use q2_min, q2_max, q2_points.
        "auto" / "physical": use 0 to physical q2max.

    q2_values:
        Discrete values, e.g. "0,1,2,q2max" or [0,1,2].

    q2_min, q2_max:
        Range endpoints. q2_max may be blank/None or string "q2max" to use physical maximum.

    q2_points:
        Number of grid points for range mode.

    validate_physical:
        Raise ValueError if any point is outside [0, (Mi-Mf)^2].

    clip_to_physical:
        If True, values outside the physical region are clipped instead of rejected.
    """
    q2_phys_max = physical_q2_max(Mi, Mf)
    mode0 = str(mode or "list").strip().lower()

    if mode0 in ("list", "lists", "discrete", "manual", "points", "set"):
        values = parse_q2_values(q2_values, q2_max=q2_phys_max)
        if not values:
            values = [0.0]

    elif mode0 in ("range", "grid", "linspace", "auto", "physical", "full"):
        qmin = 0.0 if q2_min is None or (isinstance(q2_min, float) and np.isnan(q2_min)) or str(q2_min).strip() == "" else _q2_parse_token(q2_min, q2_max=q2_phys_max)
        qmax = q2_phys_max if q2_max is None or (isinstance(q2_max, float) and np.isnan(q2_max)) or str(q2_max).strip() == "" else _q2_parse_token(q2_max, q2_max=q2_phys_max)
        n = int(q2_points)
        if n < 1:
            raise ValueError("q2_points must be >= 1.")
        if n == 1:
            values = [float(qmin)]
        else:
            values = [float(x) for x in np.linspace(float(qmin), float(qmax), n, endpoint=bool(endpoint))]

    else:
        raise ValueError("q2 mode must be 'list' or 'range'.")

    if clip_to_physical:
        values = [min(max(float(v), 0.0), q2_phys_max) for v in values]

    if validate_physical:
        bad = [v for v in values if v < -1e-12 or v > q2_phys_max + 1e-12]
        if bad:
            raise ValueError(
                f"q2 values outside physical range [0, {q2_phys_max}]: {bad}. "
                "Use a smaller range or disable validation only if you know what you are doing."
            )

    if unique:
        # preserve order while removing numerical duplicates after rounding for display safety
        seen = set()
        out = []
        for v in values:
            key = round(float(v), 12)
            if key not in seen:
                seen.add(key)
                out.append(float(v))
        values = out

    return values


def compute_form_factors_by_current_grid(
    cfg: CCQMConfig,
    final_kind: str,
    *,
    currents: list[str] | None = None,
    q2_mode: str = "list",
    q2_values=None,
    q2_min=None,
    q2_max=None,
    q2_points: int = 1,
    endpoint: bool = True,
    validate_physical: bool = True,
    include_raw_amplitudes: bool = False,
    pv_tensor_i_prefactor: bool = True,
    va_current_factor: float = 1.0,
):
    """
    Compute current-resolved form factors for a discrete set or q2 grid.

    This is the preferred backend for apps accepting user-entered q2 ranges.
    """
    grid = make_q2_grid(
        Mi=cfg.Mi,
        Mf=cfg.Mf,
        mode=q2_mode,
        q2_values=q2_values,
        q2_min=q2_min,
        q2_max=q2_max,
        q2_points=q2_points,
        endpoint=endpoint,
        validate_physical=validate_physical,
    )
    return {
        "q2_mode": q2_mode,
        "q2_values": grid,
        "q2_max_physical": physical_q2_max(cfg.Mi, cfg.Mf),
        "q2_results": {
            str(float(q2)): compute_form_factors_by_current_jsonable(
                cfg,
                final_kind,
                float(q2),
                currents=currents,
                include_raw_amplitudes=include_raw_amplitudes,
                pv_tensor_i_prefactor=pv_tensor_i_prefactor,
                va_current_factor=va_current_factor,
            )
            for q2 in grid
        },
    }

# ---------------------------------------------------------------------
# Scientific-precision diagnostics layer (v2)
# ---------------------------------------------------------------------
# This layer keeps the public API unchanged but attaches numerical diagnostics
# to every current projection.  It is intentionally appended after the validated
# fast backend so old callers continue to work.

_ORIGINAL_compute_form_factors_by_current_v2 = compute_form_factors_by_current

def _safe_condition_number(mat):
    """Return a finite condition number or inf for singular/invalid matrices."""
    try:
        arr = np.asarray(mat, dtype=complex)
        if arr.size == 0:
            return None
        c = np.linalg.cond(arr)
        if np.isfinite(c):
            return float(np.real_if_close(c))
        return float("inf")
    except Exception:
        return None


def _pp_projection_gram(p1, p2):
    P = p1 + p2
    q = p1 - p2
    return np.array([[dot(P, P), dot(P, q)], [dot(P, q), dot(q, q)]], dtype=complex)


def _pv_basis_gram(bases, p2, Mf):
    n = len(bases)
    Gmat = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            Gmat[a, b] = inner_rank2(bases[a], bases[b], p2, Mf)
    return Gmat


def _projection_diagnostics(cfg, final_kind, q2):
    """Return reusable projection diagnostics for a transition/q2 point."""
    diagnostics = {
        "q2": float(q2),
        "q2_max": float((cfg.Mi - cfg.Mf) ** 2),
        "near_q2_zero": bool(abs(q2) < 1e-10),
        "near_zero_recoil": bool(abs(float(q2) - float((cfg.Mi - cfg.Mf) ** 2)) < 1e-8),
    }
    try:
        p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)
    except Exception as exc:
        diagnostics["kinematics_error"] = str(exc)
        return diagnostics

    fk = "V" if str(final_kind).upper().startswith("V") else "P"
    if fk == "P":
        Gpp = _pp_projection_gram(p1, p2)
        diagnostics["pp_projection_condition"] = _safe_condition_number(Gpp)
        diagnostics["pp_projection_delta"] = float(np.real_if_close(np.linalg.det(Gpp))) if np.all(np.isfinite(Gpp)) else None
    else:
        try:
            Gva = _pv_basis_gram(basis_pv_va(p1, p2, cfg.Mi, cfg.Mf), p2, cfg.Mf)
            diagnostics["pv_va_projection_condition"] = _safe_condition_number(Gva)
        except Exception as exc:
            diagnostics["pv_va_projection_error"] = str(exc)
        if abs(q2) < 1e-12:
            diagnostics["pv_tensor_projection_condition"] = None
            diagnostics["pv_tensor_projection_warning"] = "P->V tensor basis contains 1/q2 at q2=0."
        else:
            try:
                Gt = _pv_basis_gram(basis_pv_tensor(p1, p2), p2, cfg.Mf)
                diagnostics["pv_tensor_projection_condition"] = _safe_condition_number(Gt)
            except Exception as exc:
                diagnostics["pv_tensor_projection_error"] = str(exc)
    return diagnostics


def _status_from_condition(cond, *, warning_threshold=1e10, danger_threshold=1e14):
    if cond is None:
        return "not_applicable"
    try:
        cond = float(cond)
    except Exception:
        return "unknown"
    if not np.isfinite(cond):
        return "singular"
    if cond >= danger_threshold:
        return "danger_ill_conditioned"
    if cond >= warning_threshold:
        return "warning_ill_conditioned"
    return "stable"


def _attach_current_diagnostics(result, cfg, final_kind, q2):
    diag = _projection_diagnostics(cfg, final_kind, q2)
    fk = "V" if str(final_kind).upper().startswith("V") else "P"
    result.setdefault("diagnostics", diag)
    for cur, entry in result.get("by_current", {}).items():
        cdiag = dict(diag)
        if fk == "P":
            if cur in ("vector", "axial", "v_minus_a", "v_plus_a", "tensor", "tensor_plus", "tensor_minus"):
                cond = diag.get("pp_projection_condition")
            else:
                cond = None
        else:
            if cur in ("tensor", "tensor_plus", "tensor_minus"):
                cond = diag.get("pv_tensor_projection_condition")
            elif cur in ("vector", "axial", "v_minus_a", "v_plus_a"):
                cond = diag.get("pv_va_projection_condition")
            else:
                cond = None
        cdiag["projection_condition_number"] = cond
        cdiag["projection_status"] = _status_from_condition(cond)
        if cdiag["projection_status"].startswith("warning") or cdiag["projection_status"].startswith("danger") or cdiag["projection_status"] == "singular":
            entry.setdefault("warning", f"Projection matrix status: {cdiag['projection_status']} (condition number {cond}).")
        entry.setdefault("diagnostics", cdiag)
    return result


def compute_form_factors_by_current(
    cfg: CCQMConfig,
    final_kind: str,
    q2: float,
    currents: list[str] | None = None,
    *,
    pv_tensor_i_prefactor: bool = True,
    va_current_factor: float = 1.0,
    include_raw_amplitudes: bool = False,
):
    """Diagnostic wrapper around the validated current-resolved form-factor engine."""
    res = _ORIGINAL_compute_form_factors_by_current_v2(
        cfg,
        final_kind,
        q2,
        currents=currents,
        pv_tensor_i_prefactor=pv_tensor_i_prefactor,
        va_current_factor=va_current_factor,
        include_raw_amplitudes=include_raw_amplitudes,
    )
    return _attach_current_diagnostics(res, cfg, final_kind, q2)


def compute_form_factors_by_current_jsonable(*args, **kwargs):
    """JSON-friendly wrapper for the diagnostic form-factor engine."""
    return _jsonable(compute_form_factors_by_current(*args, **kwargs))

# ---------------------------------------------------------------------
# Pairwise accumulation integration override
# ---------------------------------------------------------------------
# The original reference implementation accumulated the loop integral by direct
# repeated addition.  For scientific output we accumulate node contributions in
# a list and use NumPy's reduction, which uses a more stable pairwise-style
# summation path for floating arrays.  The public signature is unchanged.

def integrate_amplitude(cfg: CCQMConfig, q2: float, evaluator, shape):
    """
    Master integral with numerically stable accumulation.

      Nc g_i g_f/(4 pi^2)
      int_0^{1/lambda^2} dt t^2 int d alpha_1 d alpha_2
      exp(z-r^2/A)/A^2 * averaged_trace

    The prefactor is the checked CCQM convention used in the validation runs.
    Contributions are accumulated by vectorized reduction rather than naive
    in-place summation.
    """
    p1, p2 = kinematics(cfg.Mi, cfg.Mf, q2)

    si = 1.0 / cfg.Lambda_i ** 2
    sf = 1.0 / cfg.Lambda_f ** 2

    w13 = cfg.m3 / (cfg.m1 + cfg.m3)
    w23 = cfg.m3 / (cfg.m2 + cfg.m3)

    tmax = 1.0 / cfg.lambda_ir ** 2
    xs, ws = gauss_nodes(0.0, 1.0, cfg.n_quad)
    ts, wt = gauss_nodes(0.0, tmax, cfg.n_quad)

    terms = []
    for t, wt_i in zip(ts, wt):
        A = si + sf + t
        C = -G / (2.0 * A)  # C^{mu nu}

        for alpha1, wa1 in zip(xs, ws):
            for u, wu in zip(xs, ws):
                alpha2 = (1.0 - alpha1) * u
                alpha3 = 1.0 - alpha1 - alpha2
                jac = 1.0 - alpha1

                rho1 = si * w13 + t * alpha1
                rho2 = sf * w23 + t * alpha2

                r = rho1 * p1 + rho2 * p2

                z = (
                    si * w13 * w13 * cfg.Mi ** 2
                    + sf * w23 * w23 * cfg.Mf ** 2
                    + t * alpha1 * (cfg.Mi ** 2 - cfg.m1 ** 2)
                    + t * alpha2 * (cfg.Mf ** 2 - cfg.m2 ** 2)
                    - t * alpha3 * cfg.m3 ** 2
                )

                W = z - dot(r, r) / A
                K = -r / A

                weight = wt_i * wa1 * wu * jac * t * t * np.exp(W) / (A * A)
                terms.append(weight * evaluator(p1, p2, p1 - p2, K, C))

    if terms:
        out = np.add.reduce(np.asarray(terms, dtype=complex), axis=0)
    else:
        out = np.zeros(shape, dtype=complex)
    prefactor = cfg.Nc * cfg.g_i * cfg.g_f / (4.0 * np.pi * np.pi)
    return prefactor * out
