#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Súlykeresés Monte Carlo-val (CUDA gyorsítással, PyTorch alapokon) + Egyenletes (rácsos) súlygenerálás
# KOVARIANCIA: Excelből (hold_alapok_osszefuzve.xlsx / coverencia)
# VÁRHATÓ HOZAM (μ): HOLD stressz / kedvezőtlen / mérsékelt / kedvező forgatókönyv alapján

import argparse
import math
import random
import torch
import numpy as np
import pandas as pd

# =====================================================================
# Súlyok CSV-ből / CSV-be
# =====================================================================

def _weights_from_csv(path, assets):
    df = pd.read_csv(path)
    cols = [f"w_{a}" for a in assets]
    if all(c in df.columns for c in cols):
        W = df[cols].to_numpy()
    else:
        if all(a in df.columns for a in assets):
            W = df[assets].to_numpy()
        else:
            W = df.iloc[:, :len(assets)].to_numpy()
    return W

def _weights_to_csv(path, weights, assets):
    df = pd.DataFrame(weights, columns=[f"w_{a}" for a in assets])
    df.to_csv(path, index=False)

# =====================================================================
# KOVARIANCIA-MÁTRIX BETÖLTÉSE EXCELBŐL
# =====================================================================

def load_cov_from_excel(cov_file: str, cov_sheet: str, device: torch.device):
    df = pd.read_excel(cov_file, sheet_name=cov_sheet)

    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cov_mat = df.values.astype("float32")
    assets = list(df.columns)

    if cov_mat.shape[0] != cov_mat.shape[1]:
        raise ValueError(f"A kovariancia-mátrix nem négyzetes: {cov_mat.shape}")
    if not np.allclose(cov_mat, cov_mat.T, atol=1e-8):
        print("⚠️ Figyelmeztetés: a kovariancia-mátrix nem teljesen szimmetrikus (numerikus hiba lehet).")

    cov_annual = torch.tensor(cov_mat, device=device, dtype=torch.float32)
    return assets, cov_annual

# =====================================================================
# μ (várható éves hozamok) – több forgatókönyvvel
# =====================================================================

def build_mu_annual(assets, device, scenario: str):
    """
    Várható éves hozam (μ) beállítása a HOLD forgatókönyv adatok alapján.

    scenario: "stress" / "unfavourable" / "moderate" / "favourable"
              illetve magyar szinonimák: "stressz", "kedvezotlen",
              "mersekelt", "kedvezo".
    """

    s_raw = scenario.lower()
    if s_raw in ["stress", "stressz"]:
        s = "stress"
    elif s_raw in ["unfavourable", "kedvezotlen"]:
        s = "unfavourable"
    elif s_raw in ["favourable", "kedvezo", "kedvező"]:
        s = "favourable"
    else:
        s = "moderate"

    # százalék -> decimális: pl. -21,58% -> -0.2158
    mu_scenarios = {

        # HOLD Kötvény Befektetési Alap (HU0000702030)
        "HOLD+Kötvény+Befektetési+Alap": {
            "stress":      -0.2158,
            "unfavourable":-0.2158,
            "moderate":     0.0063,
            "favourable":   0.1808,
        },
        # HOLD VM EURO Abszolút Hozamú Alapok Alapja (HU0000708938)
        "HOLD+VM+EURO+Abszolút+Hozamú+Alapok+Alapja": {
            "stress":      -0.0466,
            "unfavourable":-0.0466,
            "moderate":     0.0001,
            "favourable":   0.0422,
        },
        # HOLD Részvény A HUF (HU0000702022)
        "HOLD+Részvény+Befektetési+Alap+A+sorozat+HUF": {
            "stress":      -0.0354,
            "unfavourable": 0.0220,
            "moderate":     0.1000,
            "favourable":   0.2194,
        },
        # HOLD Részvény B EUR (HU0000724778)
        "HOLD+Részvény+Befektetési+Alap+B+sorozat+EUR": {
            "stress":      -0.0688,
            "unfavourable":-0.0688,
            "moderate":     0.0812,
            "favourable":   0.1729,
        },
        # HOLD Galaxis EURO Abszolút Hozamú Alapok Alapja EUR (HU0000712252)
        "HOLD+Galaxis+EURO+Abszolút+Hozamú+Alapok+Alapja": {
            "stress":      -0.0330,
            "unfavourable":-0.0156,
            "moderate":     0.0467,
            "favourable":   0.1039,
        },
        # HOLD Hozamkereső Európai Származtatott Részvény Alap EUR (HU0000711916)
        "HOLD+Hozamkereső+Európai+Származtatott+Részvény+Befektetési+Alap": {
            "stress":      -0.2001,
            "unfavourable":-0.2001,
            "moderate":     0.0132,
            "favourable":   0.1325,
        },
        # HOLD Közép-európai Részvény Befektetési Alap HUF (HU0000706163)
        "HOLD+Közép-európai+Részvény+Befektetési+Alap": {
            "stress":      -0.0966,
            "unfavourable":-0.0597,
            "moderate":     0.1318,
            "favourable":   0.2923,
        },
        # HOLD Nemzetközi Részvény Alapok Alapja A HUF (HU0000702295)
        "HOLD+Nemzetközi+Részvény+Alapok+Alapja+A+sorozat": {
            "stress":      -0.1061,
            "unfavourable":-0.0148,
            "moderate":     0.1071,
            "favourable":   0.1806,
        },
        # HOLD Orion B EUR (HU0000732664)
        "HOLD+Orion+Abszolút+Hozamú+Származtatott+Befektetési+Alap+B+sorozat+EUR": {
            "stress":      -0.0583,
            "unfavourable":-0.0389,
            "moderate":     0.1033,
            "favourable":   0.1950,
        },
        # HOLD Rövid Futamidejű Kötvény Befektetési Alap HUF (HU0000701685)
        "HOLD+Rövid+Futamidejű+Kötvény+Befektetési+Alap": {
            "stress":      -0.0266,
            "unfavourable":-0.0266,
            "moderate":     0.0411,
            "favourable":   0.1304,
        },
                        # HOLD Széf USD Abszolút Hozamú Befektetési Alap
        "HOLD+Széf+USD+Abszolút+Hozamú+Befektetési+Alap": {
            "stress":       -0.0050,  # -0.50%
            "unfavourable": -0.0050,  # -0.50%
            "moderate":      0.0171,  #  1.71%
            "favourable":    0.0459,  #  4.59%
        },

        # HOLD VM Abszolút Hozamú Származtatott Befektetési Alap C sorozat (HUF)
        "HOLD+VM+Abszolút+Hozamú+Származtatott+Befektetési+Alap+C+sorozat": {
            "stress":       -0.0272,  # -2.72%
            "unfavourable": -0.0272,  # -2.72%
            "moderate":      0.0469,  #  4.69%
            "favourable":    0.1301,  # 13.01%
        },

        # Palomar Abszolút Hozamú Származtatott Befektetési Alap A sorozat HUF
        "Palomar+Abszolút+Hozamú+Származtatott+Befektetési+Alap+A+sorozat+HUF": {
            "stress":       -0.0499,  # -4.99%
            "unfavourable": -0.0499,  # -4.99%
            "moderate":      0.0564,  #  5.64%
            "favourable":    0.1687,  # 16.87%
        },

        # Palomar Abszolút Hozamú Származtatott Befektetési Alap B sorozat EUR
        "Palomar+Abszolút+Hozamú+Származtatott+Befektetési+Alap+B+sorozat+EUR": {
            "stress":       -0.0758,  # -7.58%
            "unfavourable": -0.0758,  # -7.58%
            "moderate":      0.0219,  #  2.19%
            "favourable":    0.1214,  # 12.14%
        },

        # Palomar Abszolút Hozamú Származtatott Befektetési Alap C sorozat USD
        "Palomar+Abszolút+Hozamú+Származtatott+Befektetési+Alap+C+sorozat+USD": {
            "stress":       -0.1491,  # -14.91%
            "unfavourable": -0.0660,  #  -6.60%
            "moderate":      0.0388,  #   3.88%
            "favourable":    0.1439,  #  14.39%
        },

        # Platina Delta Abszolút Hozamú Származtatott Befektetési Alap E sorozat EUR
        "Platina+Delta+Abszolút+Hozamú+Származtatott+Befektetési+Alap+E+sorozat+EUR": {
            "stress":       -0.0622,  # -6.22%
            "unfavourable": -0.0423,  # -4.23%
            "moderate":      0.1550,  # 15.50%
            "favourable":    0.3636,  # 36.36%
        },

        # Platina Delta Abszolút Hozamú Származtatott Befektetési Alap X sorozat HUF
        "Platina+Delta+Abszolút+Hozamú+Származtatott+Befektetési+Alap+X+sorozat+HUF": {
            "stress":       -0.0559,  # -5.59%
            "unfavourable": -0.0423,  # -4.23%
            "moderate":      0.1444,  # 14.44%
            "favourable":    0.3636,  # 36.36%
        },

        # Superposition Abszolút Hozamú Származtatott Befektetési Alap A sorozat HUF
        "Superposition+Abszolút+Hozamú+Származtatott+Befektetési+Alap+A+sorozat": {
            "stress":       -0.0437,  # -4.37%
            "unfavourable": -0.0327,  # -3.27%
            "moderate":      0.0703,  #  7.03%
            "favourable":    0.1822,  # 18.22%
        },

        # Superposition Abszolút Hozamú Származtatott Befektetési Alap D sorozat EUR
        "Superposition+Abszolút+Hozamú+Származtatott+Befektetési+Alap+D+sorozat+EUR": {
            "stress":       -0.0437,  # -4.37%
            "unfavourable": -0.0327,  # -3.27%
            "moderate":      0.0703,  #  7.03%
            "favourable":    0.1822,  # 18.22%
        },

        # A többi 2000/2029/3000/Beat/Columbus/Galaxis HUF/Orion A is beírható ide,
        # ha bekerülnek a kovariancia-mátrixod oszlopai közé.
    }

    mu_list = []
    for a in assets:
        if a in mu_scenarios:
            mu_list.append(mu_scenarios[a][s])
        else:
            print(f"⚠️ Nincs forgatókönyv szerinti μ ehhez az eszközhöz, 0%-ot használok: {a}")
            mu_list.append(0.0)

    mu_annual = torch.tensor(mu_list, device=device, dtype=torch.float32)
    print(f"\n📈 Várható éves hozamok (μ) – forgatókönyv: {s}:\n")
    for a, m in zip(assets, mu_list):
        print(f"  {a:80s}  {m*100:6.2f} %")
    print()
    return mu_annual

# =====================================================================
# CRN szimuláció + metrikák
# =====================================================================

def simulate_asset_growth_once(mu_annual, cov_annual, years=2.0, steps_per_year=252,
                               sims=200000, mode="gbm", seed=None, antithetic=True):
    n = mu_annual.numel()
    horizon = int(round(years * steps_per_year))
    mu_step = mu_annual / steps_per_year
    cov_step = cov_annual / steps_per_year

    eps = 1e-10
    I = torch.eye(n, device=mu_annual.device, dtype=mu_annual.dtype)
    L = torch.linalg.cholesky(cov_step + eps * I)

    if seed is not None:
        try:
            g = torch.Generator(device=mu_annual.device)
            g.manual_seed(int(seed))
        except Exception:
            torch.manual_seed(int(seed))
            g = None
    else:
        g = None

    if mode.lower() == "gbm":
        drift_step = mu_step - 0.5 * torch.diag(cov_step)
        drift_step = drift_step.view(-1, 1)
        logG = torch.zeros((n, sims), device=mu_annual.device, dtype=torch.float32)
        half = sims // 2 if (antithetic and sims % 2 == 0) else None
        for _ in range(horizon):
            if half is not None:
                Z_half = torch.randn((n, half), device=mu_annual.device, generator=g)
                Z = torch.cat([Z_half, -Z_half], dim=1)
            else:
                Z = torch.randn((n, sims), device=mu_annual.device, generator=g)
            shocks = L @ Z
            logG += drift_step + shocks
        G = torch.exp(logG)
    else:
        mu_col = mu_step.view(-1, 1)
        G = torch.ones((n, sims), device=mu_annual.device, dtype=torch.float32)
        half = sims // 2 if (antithetic and sims % 2 == 0) else None
        for _ in range(horizon):
            if half is not None:
                Z_half = torch.randn((n, half), device=mu_annual.device, generator=g)
                Z = torch.cat([Z_half, -Z_half], dim=1)
            else:
                Z = torch.randn((n, sims), device=mu_annual.device, generator=g)
            shocks = L @ Z
            r_t = mu_col + shocks
            G = G * torch.clamp(1.0 + r_t, min=1e-6)
    return G

def portfolio_metrics_from_G(W, G, riskfree_annual=0.0, years=2.0, cvar_alpha=0.05, batch_size=2048):
    K, n = W.shape
    S = G.shape[1]
    rf_total = (1.0 + riskfree_annual) ** years - 1.0

    results = []
    start = 0
    while start < K:
        end = min(start + batch_size, K)
        W_chunk = W[start:end]
        port_final = W_chunk @ G
        total_ret = port_final - 1.0

        mean_ret = total_ret.mean(dim=1)
        std_ret = total_ret.std(dim=1, unbiased=False)

        excess = mean_ret - rf_total
        sharpe = torch.where(std_ret > 0, excess / std_ret, torch.zeros_like(std_ret))

        q = torch.quantile(total_ret, cvar_alpha, dim=1, interpolation='linear')
        sorted_ret, _ = torch.sort(total_ret, dim=1)
        idx = (sorted_ret.shape[1] * cvar_alpha).ceil().clamp(min=1).to(torch.int64)
        cvars = []
        for b in range(sorted_ret.shape[0]):
            cvars.append(sorted_ret[b, :idx[b]].mean())
        cvar = torch.stack(cvars)

        chunk_out = torch.stack([mean_ret, std_ret, sharpe, q, cvar], dim=1)
        results.append(chunk_out)
        start = end

    out = torch.cat(results, dim=0)
    return out  # [mean, std, sharpe, var, cvar]

# =====================================================================
# Régi egyenkénti kiértékelés (lassú mód)
# =====================================================================

def simulate_mean_total_return(mu_annual, cov_annual, w, years=2.0, steps_per_year=252,
                               sims=50000, mode="gbm", seed=None) -> float:
    n = len(w)
    mu_step = mu_annual / steps_per_year
    cov_step = cov_annual / steps_per_year
    L = torch.linalg.cholesky(cov_step)
    if seed is not None:
        torch.manual_seed(seed)
    horizon = int(years * steps_per_year)
    Z = torch.randn(n, sims, horizon, device=w.device)
    shocks = torch.einsum('ij,jkl->ikl', L, Z)
    prices = torch.ones((n, sims), device=w.device)
    mu_col = mu_step.view(-1, 1)
    for t in range(horizon):
        r_t = mu_col + shocks[:, :, t]
        if mode == "gbm":
            prices *= torch.exp(r_t)
        else:
            prices *= (1.0 + r_t)
    port0 = torch.sum(w)
    portT = (w.view(1, -1) @ prices).view(-1)
    total_returns = portT / port0 - 1.0
    return float(total_returns.mean().cpu().item())

# =====================================================================
# Egyenletes (rácsos) súlygenerálás
# =====================================================================

def _uniform_composition(m, n, rng):
    cuts = sorted(rng.sample(range(1, m + n), n - 1))
    xs = []
    prev = 0
    for c in cuts + [m + n - 1]:
        xs.append(c - prev - 1)
        prev = c
    return xs

def sample_uniform_discrete_simplex(n_assets, step, n_samples, w_min=0.0, w_max=1.0,
                                    seed=None, max_tries_factor=200):
    """
    Egyenletes rácslépéses súlyminták a szimplexen: minden súly k*step, és a súlyok összege pontosan 1.
    KÉT ÜZEMMÓD:

    1) Ha a "m - lo*n" kicsi (0..3), akkor *direkt* legeneráljuk az ÖSSZES lehetséges rácspontot
       (csillagok-rudak, de determinisztikusan), és abból mintázunk / ismétlünk.
       -> ez a te eseted: m=20, n=19, lo=1 => R=1 => 19 darab rácspont van.

    2) Ha R nagy, a régi, lassú véletlen kompozíció helyett egyszerűen
       Dirichlet-mintákat generálunk és azokat használjuk (nem lesz tökéletesen uniform a rácson,
       de NEM fog megállni a program).
    """
    assert step > 0 and step <= 1.0, "A grid step (lépés) legyen (0,1]"
    m = int(round(1.0 / step))
    if m < 1:
        m = 1
    step = 1.0 / m
    print(f"[grid] Effective grid step = {step:.8f} (m={m})")

    # diszkrét korlátok
    lo = int(math.ceil(w_min * m - 1e-12))
    hi = int(math.floor(w_max * m + 1e-12))

    # Ha a minimum túl nagy: nincs megoldás
    if lo * n_assets > m:
        raise RuntimeError(
            f"w_min túl nagy ehhez a grid-step-hez: lo*n_assets = {lo*n_assets} > m = {m}. "
            f"Csökkentsd a w_min-t vagy növeld a step-et."
        )

    R = m - lo * n_assets  # összeg az eltolás után (y_i >= 0, sum y_i = R)

    # ------------------------
    # 1) KIS R ESETÉN: DIREKT ENUMERÁCIÓ
    # ------------------------
    if 0 <= R <= 3:
        # Generáljuk az összes nemnegatív kompozíciót: y1+...+yn = R
        compositions = []

        def gen(idx, remaining, prefix):
            if idx == n_assets - 1:
                # az utolsó kapja a maradékot
                compositions.append(prefix + [remaining])
                return
            for k in range(remaining + 1):
                gen(idx + 1, remaining - k, prefix + [k])

        gen(0, R, [])

        # Átalakítás vissza x_i = y_i + lo, és szűrés hi-re
        points = []
        for ys in compositions:
            xs = [y + lo for y in ys]
            if all(lo <= x <= hi for x in xs):
                w = [x / m for x in xs]
                points.append(w)

        if not points:
            raise RuntimeError(
                "Nem sikerült rácspontokat generálni a megadott w_min/w_max és step mellett (R kicsi, de üres a halmaz)."
            )

        points = np.array(points, dtype=float)
        print(f"[grid] Determinisztikus rács-generálás: {points.shape[0]} darab pont létezik a megadott feltételekkel.")

        # Ha kevesebb pont van, mint a kért n_samples, ismételjük / permutáljuk őket
        if points.shape[0] >= n_samples:
            # ha több van, véletlenül mintázunk közülük
            rng = np.random.default_rng(seed)
            idx = rng.choice(points.shape[0], size=n_samples, replace=False)
            return points[idx]
        else:
            # kevesebb pont van (pl. a te eseted: 19), ismételjük őket
            reps = int(math.ceil(n_samples / points.shape[0]))
            tiled = np.tile(points, (reps, 1))
            rng = np.random.default_rng(seed)
            idx = rng.permutation(tiled.shape[0])[:n_samples]
            return tiled[idx]

    # ------------------------
    # 2) NAGY R ESETÉN: GYORS DIRICHLET-ALAPÚ KÖZELÍTÉS
    # ------------------------
    print(
        f"[grid] R = m - lo*n_assets = {R}, ez már nagyobb érték, "
        f"nem enumeráljuk a teljes rácsot, hanem Dirichlet mintákat használunk."
    )
    rng = np.random.default_rng(seed)
    samples = []
    tries = 0
    need = n_samples

    while len(samples) < n_samples and tries < n_samples * 50:
        batch = rng.dirichlet(alpha=np.ones(n_assets), size=min(need, 10000))
        # korlátok
        ok = (batch >= w_min - 1e-12).all(axis=1) & (batch <= w_max + 1e-12).all(axis=1)
        batch = batch[ok]
        samples.extend(batch.tolist())
        need = n_samples - len(samples)
        tries += 1

    if len(samples) == 0:
        raise RuntimeError(
            "Dirichlet-alapú grid-közelítés sem tudott egyetlen mintát sem generálni. "
            "Próbálj kisebb w_min-t vagy lazább korlátokat."
        )
    if len(samples) < n_samples:
        samples = samples + [samples[-1]] * (n_samples - len(samples))

    return np.array(samples[:n_samples], dtype=float)


def random_weights_dirichlet(n_assets: int, n_samples: int, conc: float = 1.0,
                             w_min: float = 0.0, w_max: float = 1.0,
                             seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    tries = 0
    need = n_samples
    while len(samples) < n_samples and tries < n_samples * 50:
        batch = rng.dirichlet(alpha=np.full(n_assets, conc), size=min(need, 10000))
        ok = (batch >= w_min - 1e-12).all(axis=1) & (batch <= w_max + 1e-12).all(axis=1)
        batch = batch[ok]
        samples.extend(batch.tolist())
        need = n_samples - len(samples)
        tries += 1
    if len(samples) < n_samples and len(samples) > 0:
        samples = samples + [samples[-1]] * (n_samples - len(samples))
    return np.array(samples[:n_samples])

# =====================================================================
# Főprogram
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Futás eszköze: {device}")

    ap = argparse.ArgumentParser("CUDA-s Súlykeresés Monte Carlo-val")

    # Kovariancia forrás
    ap.add_argument("--cov-file", type=str, default="hold_alapok_osszefuzve.xlsx")
    ap.add_argument("--cov-sheet", type=str, default="coverencia")

    # Forgatókönyv választás
    ap.add_argument("--scenario", type=str, default="mersekelt",
                    help="stress / stressz / kedvezotlen / unfavourable / "
                         "mersekelt / moderate / kedvezo / favourable")

    # Szimulációs paraméterek
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--steps-per-year", type=int, default=252)
    ap.add_argument("--sims-per-eval", type=int, default=10000)
    ap.add_argument("--candidates", type=int, default=3003)

    # CRN / batch kiértékelés
    ap.add_argument("--crn", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--antithetic", action="store_true")
    ap.add_argument("--riskfree", type=float, default=0.0)
    ap.add_argument("--cvar-alpha", type=float, default=0.05)

    # Kétlépcsős kiválasztás
    ap.add_argument("--two-stage", action="store_true")
    ap.add_argument("--stage1-sims", type=int, default=10000)
    ap.add_argument("--stage1-candidates", type=int, default=None)
    ap.add_argument("--stage1-keep", type=int, default=5000)
    ap.add_argument("--stage1-out", type=str, default="stage1_rank.csv")

    # Súlygenerálás / betöltés
    ap.add_argument("--weights-csv", type=str, default=None)
    ap.add_argument("--save-weights-csv", type=str, default=None)
    ap.add_argument("--dirichlet-conc", type=float, default=1.0)
    ap.add_argument("--w-min", type=float, default=0.0)
    ap.add_argument("--w-max", type=float, default=1.0)
    ap.add_argument("--weights-method", choices=["dirichlet", "grid"], default="grid")
    ap.add_argument("--grid-step", type=float, default=0.2)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mode", choices=["gbm", "returns"], default="gbm")
    ap.add_argument("--out", default="weight_search_cuda.csv")

    args = ap.parse_args()

    # ===== Kovariancia + μ betöltése =====
    assets, cov_annual = load_cov_from_excel(args.cov_file, args.cov_sheet, device=device)
    n = len(assets)
    print(f"📊 Beolvasott eszközök száma (kovariancia): {n}")

    mu_annual = build_mu_annual(assets, device=device, scenario=args.scenario)

    # --- Súlyok előállítása / betöltése ---
    if args.weights_csv:
        weights = _weights_from_csv(args.weights_csv, assets)
    else:
        if args.weights_method == "grid":
            if args.grid_step is None:
                raise SystemExit("Kérlek add meg a --grid-step értékét (pl. --grid-step 0.01 az 1%%-os lépéshez).")
            weights = sample_uniform_discrete_simplex(
                n_assets=n,
                step=args.grid_step,
                n_samples=args.candidates,
                w_min=args.w_min,
                w_max=args.w_max,
                seed=args.seed
            )
        else:
            weights = random_weights_dirichlet(
                n_assets=n,
                n_samples=args.candidates,
                conc=args.dirichlet_conc,
                w_min=args.w_min,
                w_max=args.w_max,
                seed=args.seed
            )
        if args.save_weights_csv:
            _weights_to_csv(args.save_weights_csv, weights, assets)

    if args.weights_method == "grid" or args.no_normalize:
        W_np = weights
    else:
        W_np = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)

    # ======= Kiértékelés =======
    if args.crn and args.two_stage:
        # -------- Stage 1 --------
        K1 = args.stage1_candidates if args.stage1_candidates is not None else len(W_np)
        W1 = torch.tensor(W_np[:K1], device=device, dtype=torch.float32)

        seed1 = None if args.seed is None else int(args.seed)
        G1 = simulate_asset_growth_once(
            mu_annual, cov_annual,
            years=args.years, steps_per_year=args.steps_per_year,
            sims=args.stage1_sims, mode=args.mode, seed=seed1, antithetic=args.antithetic
        )

        M1 = portfolio_metrics_from_G(
            W1, G1,
            riskfree_annual=args.riskfree, years=args.years,
            cvar_alpha=args.cvar_alpha, batch_size=args.batch_size
        )
        df1 = pd.DataFrame({
            "mean_total_return": M1[:, 0].cpu().numpy(),
            "std_total_return":  M1[:, 1].cpu().numpy(),
            "sharpe":            M1[:, 2].cpu().numpy(),
            "VaR":               M1[:, 3].cpu().numpy(),
            "CVaR":              M1[:, 4].cpu().numpy(),
        })
        for j, a in enumerate(assets):
            df1[f"w_{a}"] = W1[:, j].cpu().numpy()
        df1 = df1.sort_values(by=["mean_total_return", "sharpe", "CVaR"],
                              ascending=[False, False, False]).reset_index(drop=True)
        df1.to_csv(args.stage1_out, index=False)

        # -------- Stage 2 --------
        K2 = min(args.stage1_keep, len(df1))
        topW = df1[[f"w_{a}" for a in assets]].iloc[:K2].to_numpy()
        W2 = torch.tensor(topW, device=device, dtype=torch.float32)

        seed2 = None if args.seed is None else int(args.seed) + 1
        G2 = simulate_asset_growth_once(
            mu_annual, cov_annual,
            years=args.years, steps_per_year=args.steps_per_year,
            sims=args.sims_per_eval, mode=args.mode, seed=seed2, antithetic=args.antithetic
        )

        M2 = portfolio_metrics_from_G(
            W2, G2,
            riskfree_annual=args.riskfree, years=args.years,
            cvar_alpha=args.cvar_alpha, batch_size=args.batch_size
        )
        df = pd.DataFrame({
            "mean_total_return": M2[:, 0].cpu().numpy(),
            "std_total_return":  M2[:, 1].cpu().numpy(),
            "sharpe":            M2[:, 2].cpu().numpy(),
            "VaR":               M2[:, 3].cpu().numpy(),
            "CVaR":              M2[:, 4].cpu().numpy(),
        })
        for j, a in enumerate(assets):
            df[f"w_{a}"] = W2[:, j].cpu().numpy()
        df = df.sort_values(by=["mean_total_return", "sharpe", "CVaR"],
                            ascending=[False, False, False]).reset_index(drop=True)

    elif args.crn:
        # -------- Egy-lépcsős CRN --------
        G = simulate_asset_growth_once(
            mu_annual, cov_annual,
            years=args.years, steps_per_year=args.steps_per_year,
            sims=args.sims_per_eval, mode=args.mode, seed=args.seed, antithetic=args.antithetic
        )
        W = torch.tensor(W_np, device=device, dtype=torch.float32)
        metrics = portfolio_metrics_from_G(
            W, G,
            riskfree_annual=args.riskfree, years=args.years,
            cvar_alpha=args.cvar_alpha, batch_size=args.batch_size
        )
        df = pd.DataFrame({
            "mean_total_return": metrics[:, 0].cpu().numpy(),
            "std_total_return":  metrics[:, 1].cpu().numpy(),
            "sharpe":            metrics[:, 2].cpu().numpy(),
            "VaR":               metrics[:, 3].cpu().numpy(),
            "CVaR":              metrics[:, 4].cpu().numpy(),
        })
        for j, a in enumerate(assets):
            df[f"w_{a}"] = W[:, j].cpu().numpy()
        df = df.sort_values(by=["mean_total_return", "sharpe", "CVaR"],
                            ascending=[False, False, False]).reset_index(drop=True)

    else:
        # -------- Régi út: jelöltenként külön szimuláció --------
        results = []
        for w_np in W_np:
            w = torch.tensor(w_np, device=device, dtype=torch.float32)
            mean_total = simulate_mean_total_return(
                mu_annual, cov_annual, w,
                years=args.years, steps_per_year=args.steps_per_year,
                sims=args.sims_per_eval, mode=args.mode, seed=args.seed
            )
            row = [mean_total] + list(w_np)
            results.append(row)
        cols = ["mean_total_return"] + [f"w_{a}" for a in assets]
        df = pd.DataFrame(results, columns=cols)
        df = df.sort_values("mean_total_return", ascending=False).reset_index(drop=True)

    df.to_csv(args.out, index=False)

    best = df.iloc[0]
    print("\n=== 🏆 Legjobb súlykészlet (átlagos végső hozam szerint) ===")
    print(f"Átlagos végső hozam ({args.years} év): {best['mean_total_return']:.4%}")
    for a in assets:
        print(f"  {a:80s}  {best['w_'+a]:.4%}")
    print(f"\nEredmények CSV: {args.out}")

if __name__ == "__main__":
    main()
