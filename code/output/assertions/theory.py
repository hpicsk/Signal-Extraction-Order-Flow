"""Checks on the steps of the paper's argument that a computation decides.

Each check states a step that qf_paper.tex prints and decides it by symbolic
algebra, so a step that a rewrite of the argument makes false fails here even
though every number of the paper still binds. The checks read neither results/ nor the
documents.

Two steps of the argument are left to the reader, because no computation
decides them for every law of the variables:

- The proof's moment step treats the signal, the noise, the return noise and
  the carrier ratio as mutually independent. The step follows from the
  independence statements of sec:theory (Remark rem:independence and the
  definitions under eq:general_dgp and eq:returns), which a computation does
  not derive; the computation below takes the independence as given and
  decides the moments.
- The proof's positivity step, that the expectation of a nonnegative
  integrand which is positive wherever omega differs from its mean is
  positive when Var(omega) > 0, is a statement about every law of omega. The
  computation decides the identity on which the step rests, not the step.
"""

from __future__ import annotations

import sympy as sp

from output.assertions.common import assert_true


# Symbols of sec:theory. The carrier ratio and turnover are positive.
_k, _gamma = sp.symbols("k gamma", positive=True)
_sig_s2, _sig_nu2, _sig_eps2 = sp.symbols("sigma_s2 sigma_nu2 sigma_eps2", positive=True)
_s, _nu, _eps = sp.symbols("s nu epsilon", real=True)
_w, _tau, _M = sp.symbols("omega tau M", positive=True)
# Moments of the carrier ratio: E[omega^j] for j = -2, -1, 1, 2.
_mw = {j: sp.Symbol(f"E_omega_{j}".replace("-", "m"), positive=True) for j in (-2, -1, 1, 2)}

_SECOND_MOMENT = {_s: _sig_s2, _nu: _sig_nu2, _eps: _sig_eps2}


def _expect(expr: sp.Expr) -> sp.Expr:
    """E[expr] for a polynomial in s, nu, epsilon and a Laurent polynomial in
    omega, with the four mutually independent, s, nu and epsilon of mean zero
    and of the variances of sec:theory."""
    total = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff, value = sp.Integer(1), sp.Integer(1)
        for base, power in term.as_powers_dict().items():
            if base in _SECOND_MOMENT:
                if power == 1:
                    value = sp.Integer(0)
                elif power == 2:
                    value *= _SECOND_MOMENT[base]
                else:
                    raise ValueError(f"moment {base}^{power} is not one the argument uses")
            elif base == _w:
                value *= _mw[int(power)]
            else:
                coeff *= base ** power
        total += coeff * value
    return sp.simplify(total)


def _cov(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return sp.simplify(_expect(a * b) - _expect(a) * _expect(b))


def _same(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    return sp.simplify(sp.expand(lhs - rhs)) == 0


def check_argument_steps(_m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:theory (eq:mismatched, prop:matched_filter,
    eq:threshold, cor:regimes, the factor sentence after
    prop:matched_filter), subsec:threshold_estimate,
    subsec:proof_matched_filter (eq:snr_matched, eq:snr_mismatched),
    subsec:return_prediction, eq:reversal_contrast, subsec:free_exponent,
    subsec:appendix_adv_normalisation and eq:proportionality.

    Every step below is one that the paper prints, decided by symbolic
    algebra on the model of sec:theory."""
    k, g, ss, sn, se = _k, _gamma, _sig_s2, _sig_nu2, _sig_eps2
    s, nu, eps, w, tau, M = _s, _nu, _eps, _w, _tau, _M
    m1, m2, mm2 = _mw[1], _mw[2], _mw[-2]
    R = g * s + eps
    sig_R2 = g**2 * ss + se

    # eq:mismatched and the matched filter of sec:theory: with omega = G/H,
    # D/G = k s + nu H/G and D/H = k s omega + nu.
    G, H = sp.symbols("G H", positive=True)
    D = k * s * G + nu * H
    assert_true(_same(D / G, k * s + nu * H / G) and _same(D / H, k * s * G / H + nu),
                claim="eq:mismatched: S^matched = D/G = k s + nu H/G and "
                      "S^mismatched = D/H = k s omega + nu with omega = G/H")

    # sec:theory: the four filters of the two regimes, with V = tau M.
    V = tau * M
    D_cap, D_vol = k * s * M + nu * V, k * s * V + nu * M
    assert_true(
        _same(D_cap / M, k * s + nu * tau) and _same(D_cap / V, k * s / tau + nu)
        and _same(D_vol / V, k * s + nu / tau) and _same(D_vol / M, k * s * tau + nu),
        claim="sec:theory: S^MC = ks + nu tau and S^TV = ks tau^-1 + nu in the capacity "
              "regime, S^TV = ks + nu tau^-1 and S^MC = ks tau + nu in the volume regime")
    # cor:regimes: (G, H) = (M, V) sets omega = 1/tau and (V, M) sets omega = tau.
    assert_true(_same(M / V, 1 / tau) and _same(V / M, tau),
                claim="cor:regimes: (G,H) = (M,V) and (V,M) set omega = tau^-1 and tau")

    # sec:theory: the return noise is independent of the flow, so
    # rho(S,R) = rho(S,s) rho(s,R) for either filter.
    for S_f in (k * s + nu / w, k * s * w + nu):
        rho_SR2 = _cov(S_f, g * s + eps) ** 2 / (_cov(S_f, S_f) * (g**2 * ss + se))
        rho_Ss2 = _cov(S_f, s) ** 2 / (_cov(S_f, S_f) * ss)
        rho_sR2 = (g * ss) ** 2 / (ss * (g**2 * ss + se))
        assert_true(_same(rho_SR2, rho_Ss2 * rho_sR2),
                    claim="sec:theory: rho(S,R) = rho(S,s) rho(s,R), and rho(s,R) is the "
                          "same for both filters")

    # sec:theory: the squared correlation rises with rho^2 / (1 - rho^2).
    r2 = sp.Symbol("rho2", positive=True)
    assert_true(_same(sp.diff(r2 / (1 - r2), r2), 1 / (1 - r2) ** 2),
                claim="sec:theory: rho^2(S,s) rises with the output signal-to-noise "
                      "ratio rho^2(S,s)/(1-rho^2(S,s))")

    # subsec:proof_matched_filter: the moments of the two filters.
    S_m, S_x = k * s + nu / w, k * s * w + nu
    cov_m, var_m = _cov(S_m, R), _cov(S_m, S_m)
    cov_x, var_x = _cov(S_x, R), _cov(S_x, S_x)
    assert_true(_same(cov_m, k * g * ss) and _same(var_m, k**2 * ss + sn * mm2),
                claim="subsec:proof_matched_filter: Cov(S^matched, R) = k gamma sigma_s^2 "
                      "and Var(S^matched) = k^2 sigma_s^2 + sigma_nu^2 E[omega^-2]",
                detail=f"computed Cov = {cov_m}, Var = {var_m}")
    assert_true(_same(cov_x, k * g * ss * m1) and _same(var_x, k**2 * ss * m2 + sn),
                claim="subsec:proof_matched_filter: Cov(S^mismatched, R) = k gamma sigma_s^2 "
                      "E[omega] and Var(S^mismatched) = k^2 sigma_s^2 E[omega^2] + sigma_nu^2",
                detail=f"computed Cov = {cov_x}, Var = {var_x}")
    assert_true(_same(_cov(R, R), sig_R2),
                claim="subsec:proof_matched_filter: sigma_R^2 = gamma^2 sigma_s^2 + sigma_eps^2")
    assert_true(bool(cov_m.is_positive) and bool(cov_x.is_positive),
                claim="subsec:proof_matched_filter: both return covariances are positive")

    # eq:snr_matched and eq:snr_mismatched, and the cross-multiplication: the
    # difference of the squared correlations has the sign of B - E[omega]^2 A.
    rho2_m = cov_m**2 / (var_m * sig_R2)
    rho2_x = cov_x**2 / (var_x * sig_R2)
    A, B = k**2 * ss + sn * mm2, k**2 * ss * m2 + sn
    assert_true(
        _same((rho2_m - rho2_x) * A * B * sig_R2 / (k * g * ss) ** 2, B - m1**2 * A),
        claim="subsec:proof_matched_filter: rho^2_matched > rho^2_mismatched exactly when "
              "k^2 sigma_s^2 E[omega^2] + sigma_nu^2 > (E[omega])^2 (k^2 sigma_s^2 + "
              "sigma_nu^2 E[omega^-2])")
    # Collecting terms with the variance identity, then dividing by
    # sigma_nu^2 Var(omega), gives eq:threshold.
    var_w = m2 - m1**2
    phi = (m1**2 * mm2 - 1) / var_w
    assert_true(_same(B - m1**2 * A, k**2 * ss * var_w - sn * (m1**2 * mm2 - 1)),
                claim="subsec:proof_matched_filter: collecting terms with Var(omega) = "
                      "E[omega^2] - (E[omega])^2 gives k^2 sigma_s^2 Var(omega) > "
                      "sigma_nu^2 ((E[omega])^2 E[omega^-2] - 1)")
    assert_true(_same((B - m1**2 * A) / (sn * var_w), k**2 * ss / sn - phi),
                claim="eq:threshold: the condition is SNR > Phi(omega) = "
                      "((E[omega])^2 E[omega^-2] - 1) / Var(omega)")

    # The factor sentence after prop:matched_filter: scaled to the matched
    # filter's covariance, the mismatched filter's signal variance is
    # E[omega^2]/(E[omega])^2 times the matched filter's, and the matched
    # filter's noise variance (E[omega])^2 E[omega^-2] times the mismatched
    # filter's; the first factor exceeds one by Var(omega)/(E[omega])^2.
    # ... so that the filter with the smaller variance has the larger
    # correlation: at a fixed covariance C the correlation C/(sqrt(V) sigma_R)
    # falls as the variance V rises.
    C, Vs, sR = sp.symbols("C V sigma_R", positive=True)
    assert_true(bool(sp.diff(C / (sp.sqrt(Vs) * sR), Vs).is_negative),
                claim="sec:theory: at the same covariance with returns, the filter with "
                      "the smaller variance has the larger correlation")
    sig_var_x, noise_var_x = k**2 * ss * m2 / m1**2, sn / m1**2
    assert_true(_same(sig_var_x / (k**2 * ss), m2 / m1**2)
                and _same((sn * mm2) / noise_var_x, m1**2 * mm2)
                and _same(m2 / m1**2 - 1, var_w / m1**2),
                claim="sec:theory: once scaled to the same covariance with returns, the "
                      "signal-variance factor is E[omega^2]/(E[omega])^2 and the "
                      "noise-variance factor is (E[omega])^2 E[omega^-2]")

    # subsec:proof_matched_filter: the identity, for positive x and m, on which
    # (E[omega])^2 E[omega^-2] > 1 rests. Averaging it at m = E[omega] removes
    # the linear term.
    x, mu = sp.symbols("x m", positive=True)
    rem = (x - mu) ** 2 * (2 * x + mu) / (x**2 * mu**3)
    assert_true(_same(x**-2, mu**-2 - 2 * (x - mu) / mu**3 + rem),
                claim="subsec:proof_matched_filter: x^-2 = m^-2 - 2 m^-3 (x - m) + "
                      "(x - m)^2 (2x + m) / (x^2 m^3) for positive x and m")
    assert_true(_same(_expect(w - m1).subs(_mw[1], m1), 0),
                claim="subsec:proof_matched_filter: the linear term averages to zero at "
                      "m = E[omega]")

    # subsec:threshold_estimate, subsec:appendix_j_rank and
    # subsec:appendix_participation_matching: S^MC = tau S^TV.
    Dn = sp.Symbol("D", real=True)
    assert_true(_same(Dn / M, tau * (Dn / V)),
                claim="subsec:threshold_estimate: S^MC = tau S^TV identically")

    # subsec:threshold_estimate and subsec:free_exponent: the family
    # D / (M^(1-theta) V^theta) equals S^MC tau^-theta, and a common factor c
    # on D, M and V leaves D / (M^a V^b) unchanged only when a + b = 1.
    th, a, b, c = sp.symbols("theta a b c", positive=True)
    fam = Dn / (M ** (1 - th) * V**th)
    assert_true(_same(sp.powsimp(fam / ((Dn / M) * tau**-th), force=True), 1)
                and _same(fam.subs(th, 0), Dn / M) and _same(fam.subs(th, 1), Dn / V),
                claim="subsec:free_exponent: D/(M^(1-theta) V^theta) = S^MC tau^-theta, "
                      "with S^MC at theta = 0 and S^TV at theta = 1")
    gen = Dn / (M**a * V**b)
    ratio = sp.powsimp(sp.expand_power_base(
        (c * Dn) / ((c * M) ** a * (c * tau * M) ** b) / gen, force=True), force=True)
    assert_true(_same(ratio, c ** (1 - a - b)),
                claim="subsec:threshold_estimate: scaling a firm and its flow by c leaves "
                      "D/(M^a V^b) unchanged only when the exponents sum to one",
                detail=f"computed ratio {ratio}")

    # subsec:return_prediction: for a regressor of unit variance the
    # univariate slope is its correlation with returns times the returns'
    # standard deviation.
    cxr, sdr = sp.symbols("C_xR sigma_R", positive=True)
    slope, rho = cxr / 1, cxr / (1 * sdr)
    assert_true(_same(slope, rho * sdr),
                claim="subsec:return_prediction: for a regressor standardized within the "
                      "day, the univariate slope equals its correlation with returns times "
                      "the standard deviation of the day's returns")
    # sec:theory, Economic interpretation: fitted values a + bS correlate with
    # returns exactly as strongly as S.
    a0, b0, vS, vR, cSR = sp.symbols("a0 b0 V_S V_R C_SR", real=True)
    assert_true(_same((b0 * cSR) ** 2 / (b0**2 * vS * vR), cSR**2 / (vS * vR)),
                claim="sec:theory: fitted values that are an affine function of the "
                      "mismatched filter correlate with returns exactly as strongly as "
                      "the filter")

    # eq:reversal_contrast.
    b1, b2, b3, b4 = sp.symbols("b1 b2 b3 b4", real=True)
    assert_true(_same((b1 - b2) - ((b1 + b3) - (b2 + b4)), b4 - b3),
                claim="eq:reversal_contrast: (b1 - b2) - [(b1 + b3) - (b2 + b4)] = b4 - b3")

    # subsec:appendix_adv_normalisation: D / ADV = S^MC / tau_bar with
    # tau_bar = ADV / M.
    adv = sp.Symbol("ADV", positive=True)
    assert_true(_same(Dn / adv, (Dn / M) / (adv / M)),
                claim="subsec:appendix_adv_normalisation: the rebuilt normalized flow "
                      "equals S^MC / tau_bar with tau_bar = ADV/M")

    # eq:proportionality: b_f = kappa q_f for both filters gives
    # b_MC q_TV - b_TV q_MC = 0 for any kappa.
    kap, qmc, qtv = sp.symbols("kappa q_MC q_TV", real=True)
    assert_true(_same((kap * qmc) * qtv - (kap * qtv) * qmc, 0),
                claim="eq:proportionality: b_MC q_TV - b_TV q_MC = 0 for any impact "
                      "coefficient")
