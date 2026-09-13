#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neurális portfólió-optimalizáló Monte Carlo-val + PID szabályozással.

Kovariancia + μ:
    - Kovariancia-mátrix: hold_alapok_osszefuzve.xlsx / coverencia (vagy ami meg van adva)
    - Várható hozam (μ): HOLD stressz / kedvezőtlen / mérsékelt / kedvező forgatókönyv alapján,
      ugyanazzal a build_mu_annual() függvénnyel, mint a cuda_crn_twostage_grid-ben.

Alapértelmezett loss komponensek (ha nincs argumentum):
    - MEAN: várható hozam
    - MEDIAN: medián hozam
    - Sharpe: mean / std
    - Entropy: diverzifikáció

További opcionális loss elemek:
    - CVaR: tail risk
    - Max weight penalty: koncentráció büntetése

PID:
    - metric PID: egy választott metrikára (Sharpe / mean / median / CVaR)
    - loss PID: magára a loss-ra
"""

import argparse
import torch
import pandas as pd

from cuda_crn_twostage_grid import (
    load_cov_from_excel,
    build_mu_annual,
    simulate_asset_growth_once,
    simulate_mean_total_return,
)

# ---------------------------------------------------------------------
# Risk / stat eszközök
# ---------------------------------------------------------------------

def compute_cvar(total_ret: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    CVaR_alpha (Conditional Value-at-Risk) becslés.
    total_ret: (sims,) hozamvektor
    alpha: alsó kvantilis (pl. 0.05 = legrosszabb 5%)
    """
    sims = total_ret.numel()
    if sims == 0:
        return torch.tensor(0.0, device=total_ret.device)

    k = max(1, int(alpha * sims))
    sorted_ret, _ = total_ret.sort()  # növekvő sorrend
    tail = sorted_ret[:k]
    return tail.mean()  # diszkrét tail-átlag

def build_loss(
    mean_ret: torch.Tensor,
    median_ret: torch.Tensor,
    std_ret: torch.Tensor,
    cvar: torch.Tensor,
    entropy: torch.Tensor,
    penalty_max: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Loss modulárisan felépítve a cfg beállításai alapján.
    """
    eps = 1e-8
    loss = 0.0 * mean_ret  # tensor legyen (autograd miatt)

    # 0) Várható hozam – maximalizáljuk
    if cfg.use_mean:
        loss = loss - cfg.lambda_mean * mean_ret

    # 0/b) Medián hozam – maximalizáljuk
    if cfg.use_median:
        loss = loss - cfg.lambda_median * median_ret

    # 1) Sharpe-szerű metrika – maximalizáljuk
    if cfg.use_sharpe:
        metric = mean_ret / (std_ret + eps)
        loss = loss - cfg.lambda_sharpe * metric

    # 2) CVaR – tail risk (minél kevésbé negatív annál jobb)
    if cfg.use_cvar:
        risk_term_cvar = -cvar      # nagy bukó → nagy pozitív szám
        loss = loss + cfg.lambda_cvar * risk_term_cvar

    # 3) Entropy – diverzifikációt maximalizáljuk
    if cfg.use_entropy:
        loss = loss - cfg.lambda_ent * entropy

    # 4) Max weight penalty – koncentráció büntetése
    if cfg.use_maxweight:
        loss = loss + cfg.lambda_max * penalty_max

    return loss

# ---------------------------------------------------------------------
# NN modell
# ---------------------------------------------------------------------

class NeuralPortfolio(torch.nn.Module):
    """Egyszerű MLP, ami súlyvektort javasol a portfólióra."""

    def __init__(self, n_assets, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        in_dim = n_assets
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(torch.nn.ReLU())
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, n_assets))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------
# PID segédfüggvény
# ---------------------------------------------------------------------

def pid_update(kp, ki, kd, error, error_int, error_prev, dt=1.0):
    """
    Klasszikus PID:
        u = Kp * e + Ki * ∫e dt + Kd * de/dt
    """
    error_int_new = error_int + error * dt
    d_error = (error - error_prev) / dt
    u = kp * error + ki * error_int_new + kd * d_error
    return u, error_int_new, error

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train_neural_portfolio(
    mu_annual,
    cov_annual,
    years=2.0,
    steps_per_year=252,
    sims=500000,
    epochs=5000,
    base_lr=1e-3,
    alpha_cvar=0.05,
    # loss lambdák
    lambda_mean=1.0,
    lambda_median=1.0,
    lambda_sharpe=1.0,
    lambda_cvar=0.0,
    lambda_ent=0.00,
    lambda_max=0.0,
    # max weight korlát
    max_weight=1.0,
    # loss komponensek kapcsolása
    use_mean=True,
    use_median=True,
    use_sharpe=True,
    use_cvar=False,
    use_entropy=False,
    use_maxweight=False,
    # PID beállítások
    use_pid_metric=True,
    use_pid_loss=False,
    metric_source="mean",   # "sharpe" / "mean" / "median" / "cvar"
    target_metric=1.0,
    target_loss=0.0,
    kp_metric=0.2,
    ki_metric=0.0,
    kd_metric=0.05,
    kp_loss=0.0,
    ki_loss=0.0,
    kd_loss=0.0,
    scale_min=0.1,
    scale_max=10.0,
    mode="gbm",
    seed=123,
):
    """
    Neurális portfólió tanítása moduláris loss-szal + PID-del.
    """

    device_local = mu_annual.device
    n_assets = mu_annual.numel()

    # Fix CRN: egyetlen sztochasztikus pályahalmaz, ezen tanulunk
    G = simulate_asset_growth_once(
        mu_annual,
        cov_annual,
        years=years,
        steps_per_year=steps_per_year,
        sims=sims,
        mode=mode,
        seed=seed,
        antithetic=True,
    )  # (n_assets, sims)

    # Konfig objektum
    class Cfg:
        pass

    cfg = Cfg()
    cfg.use_mean = use_mean
    cfg.use_median = use_median
    cfg.use_sharpe = use_sharpe
    cfg.use_cvar = use_cvar
    cfg.use_entropy = use_entropy
    cfg.use_maxweight = use_maxweight

    cfg.lambda_mean = lambda_mean
    cfg.lambda_median = lambda_median
    cfg.lambda_sharpe = lambda_sharpe
    cfg.lambda_cvar = lambda_cvar
    cfg.lambda_ent = lambda_ent
    cfg.lambda_max = lambda_max

    cfg.max_weight = max_weight

    cfg.use_pid_metric = use_pid_metric
    cfg.use_pid_loss = use_pid_loss
    cfg.metric_source = metric_source
    cfg.target_metric = target_metric
    cfg.target_loss = target_loss

    cfg.kp_metric = kp_metric
    cfg.ki_metric = ki_metric
    cfg.kd_metric = kd_metric

    cfg.kp_loss = kp_loss
    cfg.ki_loss = ki_loss
    cfg.kd_loss = kd_loss

    cfg.scale_min = scale_min
    cfg.scale_max = scale_max

    model = NeuralPortfolio(n_assets=n_assets).to(device_local)
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    x_dummy = torch.ones((1, n_assets), device=device_local)

    # PID állapotok
    error_int_metric = 0.0
    error_prev_metric = 0.0
    error_int_loss = 0.0
    error_prev_loss = 0.0

    best_metric_val = -1e9
    best_w = None
    history = []

    eps = 1e-8

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        logits = model(x_dummy)
        w = torch.softmax(logits, dim=-1).view(-1)  # (n_assets,)

        # Portfólió hozamok
        port_final = (w.unsqueeze(0) @ G).squeeze(0)
        total_ret = port_final - 1.0

        mean_ret = total_ret.mean()
        median_ret = total_ret.median()
        std_ret = total_ret.std(unbiased=False)
        entropy = -(w * (w + eps).log()).sum()
        cvar = compute_cvar(total_ret, alpha=alpha_cvar)

        over = torch.relu(w - cfg.max_weight)
        penalty_max = (over ** 2).sum()

        # Loss felépítése
        loss = build_loss(mean_ret, median_ret, std_ret, cvar, entropy, penalty_max, cfg)

        # ----- PID metrika kiválasztás -----
        with torch.no_grad():
            if cfg.metric_source == "sharpe":
                metric_value = float((mean_ret / (std_ret + eps)).cpu())
            elif cfg.metric_source == "mean":
                metric_value = float(mean_ret.cpu())
            elif cfg.metric_source == "median":
                metric_value = float(median_ret.cpu())
            elif cfg.metric_source == "cvar":
                metric_value = float(cvar.cpu())
            else:
                metric_value = float((mean_ret / (std_ret + eps)).cpu())

            loss_value = float(loss.detach().cpu())

        # ----- PID számítás -----
        u_metric = 0.0
        if cfg.use_pid_metric:
            error_metric = cfg.target_metric - metric_value
            u_metric, error_int_metric, error_prev_metric = pid_update(
                cfg.kp_metric, cfg.ki_metric, cfg.kd_metric,
                error_metric, error_int_metric, error_prev_metric, dt=1.0
            )

        u_loss = 0.0
        if cfg.use_pid_loss:
            error_loss = cfg.target_loss - loss_value
            u_loss, error_int_loss, error_prev_loss = pid_update(
                cfg.kp_loss, cfg.ki_loss, cfg.kd_loss,
                error_loss, error_int_loss, error_prev_loss, dt=1.0
            )

        u_total = u_metric + u_loss
        scale = 1.0 + u_total
        scale = max(cfg.scale_min, min(cfg.scale_max, scale))
        scale_tensor = torch.tensor(scale, device=device_local, dtype=loss.dtype)

        # Loss skálázása PID-dal
        loss_scaled = loss * scale_tensor
        loss_scaled.backward()
        opt.step()

        # Log értékek
        mean_ret_value = float(mean_ret.detach().cpu())
        median_ret_value = float(median_ret.detach().cpu())
        std_ret_value = float(std_ret.detach().cpu())
        cvar_value = float(cvar.detach().cpu())
        entropy_value = float(entropy.detach().cpu())
        penalty_max_value = float(penalty_max.detach().cpu())

        if metric_value > best_metric_val:
            best_metric_val = metric_value
            best_w = w.detach().cpu()

        history.append(
            (
                epoch,
                mean_ret_value,
                median_ret_value,
                std_ret_value,
                cvar_value,
                entropy_value,
                penalty_max_value,
                metric_value,
                loss_value,
                u_metric,
                u_loss,
                scale,
            )
        )

        if epoch % max(1, epochs // 10) == 0:
            print(
                f"[NN train] Epoch {epoch}/{epochs}  "
                f"mean={mean_ret_value:.4%}  median={median_ret_value:.4%}  std={std_ret_value:.4%}  "
                f"CVaR={cvar_value:.4%}  Ent={entropy_value:.3f}  "
                f"MaxPen={penalty_max_value:.6f}  "
                f"Metric={metric_value:.4f}  Loss={loss_value:.4f}  "
                f"u_m={u_metric:.3f} u_l={u_loss:.3f} scale={scale:.3f}"
            )

    return best_w, history

# ---------------------------------------------------------------------
# main – argparse, de alapból mean+median+sharpe+entropy
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("NN portfólió-optimalizáló (mean+median+Sharpe+entropy alapból)")

    # Kovariancia / μ forrás – ugyanaz, mint a grid-es kódban
    ap.add_argument("--cov-file", type=str, default="hold_alapok_osszefuzve.xlsx")
    ap.add_argument("--cov-sheet", type=str, default="coverencia")
    ap.add_argument(
        "--scenario",
        type=str,
        default="favourable",
        help="stress / stressz / kedvezotlen / unfavourable / "
             "mersekelt / moderate / kedvezo / favourable",
    )

    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--steps-per-year", type=int, default=252)
    ap.add_argument("--nn-sims", type=int, default=150000)
    ap.add_argument("--sims-per-eval", type=int, default=100)

    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--alpha-cvar", type=float, default=0.05)
    ap.add_argument("--max-weight", type=float, default=1.0)

    # Loss komponensek kapcsolása – ALAP: mean+median+sharpe+entropy ON
    ap.add_argument("--use-mean",   type=int, choices=[0, 1], default=1)
    ap.add_argument("--use-median", type=int, choices=[0, 1], default=1)
    ap.add_argument("--use-sharpe", type=int, choices=[0, 1], default=1)
    ap.add_argument("--use-cvar",   type=int, choices=[0, 1], default=0)
    ap.add_argument("--use-entropy", type=int, choices=[0, 1], default=0)
    ap.add_argument("--use-maxweight", type=int, choices=[0, 1], default=0)

    # Lambda súlyok
    ap.add_argument("--lambda-mean",   type=float, default=1.0)
    ap.add_argument("--lambda-median", type=float, default=1.0)
    ap.add_argument("--lambda-sharpe", type=float, default=1.0)
    ap.add_argument("--lambda-cvar",   type=float, default=0.0)
    ap.add_argument("--lambda-ent",    type=float, default=0.00)
    ap.add_argument("--lambda-max",    type=float, default=0.0)

    # PID kapcsolók – ALAP: metric PID ON, loss PID ON
    ap.add_argument("--use-pid-metric", type=int, choices=[0, 1], default=1)
    ap.add_argument("--use-pid-loss",   type=int, choices=[0, 1], default=1)

    # PID forrás/metrika
    ap.add_argument("--metric-source", choices=["sharpe", "mean", "median", "cvar"], default="mean")
    ap.add_argument("--target-metric", type=float, default=1.0)
    ap.add_argument("--target-loss",   type=float, default=20.0)

    # PID gain-ek (metric PID)
    ap.add_argument("--kp-metric", type=float, default=0.2)
    ap.add_argument("--ki-metric", type=float, default=0.0)
    ap.add_argument("--kd-metric", type=float, default=0.05)

    # PID gain-ek (loss PID)
    ap.add_argument("--kp-loss", type=float, default=0.1)
    ap.add_argument("--ki-loss", type=float, default=0.0)
    ap.add_argument("--kd-loss", type=float, default=0.02)

    # PID scale clamp
    ap.add_argument("--scale-min", type=float, default=0.1)
    ap.add_argument("--scale-max", type=float, default=10.0)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mode", choices=["gbm", "returns"], default="gbm")
    ap.add_argument("--out", type=str, default="nn_portfolio_modular_pid.csv")

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Futás eszköze: {device}")

    # Kovariancia + μ beolvasása ugyanúgy, mint a grid-es kódban
    assets, cov_annual = load_cov_from_excel(args.cov_file, args.cov_sheet, device=device)
    mu_annual = build_mu_annual(assets, device=device, scenario=args.scenario)

    print(
        "🧠 NN portfólió-optimalizáló indul.\n"
        f"   years={args.years}, steps/year={args.steps_per_year}, sims={args.nn_sims}, epochs={args.epochs}\n"
        f"   scenario={args.scenario}\n"
        f"   loss: use_mean={args.use_mean}, use_median={args.use_median}, "
        f"use_sharpe={args.use_sharpe}, use_cvar={args.use_cvar}, "
        f"use_entropy={args.use_entropy}, use_maxweight={args.use_maxweight}\n"
        f"   λ_mean={args.lambda_mean}, λ_median={args.lambda_median}, λ_sharpe={args.lambda_sharpe}, "
        f"λ_cvar={args.lambda_cvar}, λ_ent={args.lambda_ent}, λ_max={args.lambda_max}, "
        f"max_weight={args.max_weight}\n"
        f"   PID_metric: use={args.use_pid_metric}, source={args.metric_source}, "
        f"target_metric={args.target_metric}, Kp={args.kp_metric}, "
        f"Ki={args.ki_metric}, Kd={args.kd_metric}\n"
        f"   PID_loss: use={args.use_pid_loss}, target_loss={args.target_loss}, "
        f"Kp={args.kp_loss}, Ki={args.ki_loss}, Kd={args.kd_loss}\n"
        f"   scale ∈ [{args.scale_min}, {args.scale_max}]"
    )

    best_w_cpu, history = train_neural_portfolio(
        mu_annual,
        cov_annual,
        years=args.years,
        steps_per_year=args.steps_per_year,
        sims=args.nn_sims,
        epochs=args.epochs,
        base_lr=args.base_lr,
        alpha_cvar=args.alpha_cvar,
        lambda_mean=args.lambda_mean,
        lambda_median=args.lambda_median,
        lambda_sharpe=args.lambda_sharpe,
        lambda_cvar=args.lambda_cvar,
        lambda_ent=args.lambda_ent,
        lambda_max=args.lambda_max,
        max_weight=args.max_weight,
        use_mean=bool(args.use_mean),
        use_median=bool(args.use_median),
        use_sharpe=bool(args.use_sharpe),
        use_cvar=bool(args.use_cvar),
        use_entropy=bool(args.use_entropy),
        use_maxweight=bool(args.use_maxweight),
        use_pid_metric=bool(args.use_pid_metric),
        use_pid_loss=bool(args.use_pid_loss),
        metric_source=args.metric_source,
        target_metric=args.target_metric,
        target_loss=args.target_loss,
        kp_metric=args.kp_metric,
        ki_metric=args.ki_metric,
        kd_metric=args.kd_metric,
        kp_loss=args.kp_loss,
        ki_loss=args.ki_loss,
        kd_loss=args.kd_loss,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        mode=args.mode,
        seed=args.seed,
    )

    best_w = best_w_cpu.to(device)

    mean_total = simulate_mean_total_return(
        mu_annual,
        cov_annual,
        best_w,
        years=args.years,
        steps_per_year=args.steps_per_year,
        sims=args.sims_per_eval,
        mode=args.mode,
        seed=args.seed,
    )

    print("\n=== 🧠 NN-optimalizált súlyok ===")
    print(f"Átlagos végső hozam ({args.years} év): {mean_total:.4%}")
    for a, w_i in zip(assets, best_w_cpu.numpy()):
        print(f"  {a:80s}  {w_i:.4%}")

    df = pd.DataFrame(
        [[mean_total] + list(best_w_cpu.numpy())],
        columns=["mean_total_return"] + [f"w_{a}" for a in assets],
    )
    df.to_csv(args.out, index=False)
    print(f"\nEredmények CSV: {args.out}")

if __name__ == "__main__":
    main()
