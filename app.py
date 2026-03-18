"""
Autocallable Structured Product — S&P 500
Web application with interactive pricing dashboard.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from dataclasses import dataclass, asdict

app = Flask(__name__)


@dataclass
class AutocallableParams:
    spot: float
    strike: float
    autocall_barrier: float
    coupon_barrier: float
    put_barrier: float
    coupon_rate: float
    notional: float
    maturity_years: float
    obs_freq: int
    memory_coupon: bool


@dataclass
class MarketParams:
    risk_free_rate: float
    dividend_yield: float
    volatility: float


def simulate_paths(market, spot, maturity, n_steps, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    drift = (market.risk_free_rate - market.dividend_yield - 0.5 * market.volatility**2) * dt
    diffusion = market.volatility * np.sqrt(dt)
    log_returns = drift + diffusion * rng.standard_normal((n_paths, n_steps))
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1)
    return spot * np.exp(log_paths)


def price_autocallable(ac, market, n_paths=100_000, seed=42):
    n_obs = int(ac.maturity_years * ac.obs_freq)
    total_steps = int(ac.maturity_years * 252)
    paths = simulate_paths(market, ac.spot, ac.maturity_years, total_steps, n_paths, seed)

    obs_indices = [int(round(i * total_steps / n_obs)) for i in range(1, n_obs + 1)]
    obs_times = [i * ac.maturity_years / n_obs for i in range(1, n_obs + 1)]

    autocall_level = ac.autocall_barrier * ac.strike
    coupon_level = ac.coupon_barrier * ac.strike
    put_level = ac.put_barrier * ac.strike
    coupon_amount = ac.coupon_rate * ac.notional

    payoffs = np.zeros(n_paths)
    redemption_times = np.full(n_paths, ac.maturity_years)
    called = np.zeros(n_paths, dtype=bool)
    path_min = np.min(paths, axis=1)
    knocked_in = path_min < put_level
    missed_coupons = np.zeros(n_paths)

    # Track autocall counts per observation
    autocall_counts = []

    for k, (obs_idx, obs_t) in enumerate(zip(obs_indices, obs_times)):
        spot_at_obs = paths[:, obs_idx]
        active = ~called
        below_coupon = active & (spot_at_obs < coupon_level)
        if ac.memory_coupon:
            missed_coupons[below_coupon] += coupon_amount

        autocalled_now = active & (spot_at_obs >= autocall_level)
        autocall_counts.append(int(autocalled_now.sum()))

        if autocalled_now.any():
            earned = coupon_amount
            if ac.memory_coupon:
                earned = earned + missed_coupons[autocalled_now]
            payoffs[autocalled_now] = ac.notional + earned
            redemption_times[autocalled_now] = obs_t
            called[autocalled_now] = True
            if ac.memory_coupon:
                missed_coupons[autocalled_now] = 0.0

    alive = ~called
    if alive.any():
        final_spot = paths[alive, -1]
        mat_payoff = np.zeros(alive.sum())
        above_coupon_mat = final_spot >= coupon_level
        below_put_mat = knocked_in[alive] & (final_spot < ac.strike)
        principal_only = ~above_coupon_mat & ~below_put_mat

        coupon_pay = coupon_amount
        if ac.memory_coupon:
            coupon_pay = coupon_amount + missed_coupons[alive]
        mat_payoff[above_coupon_mat] = ac.notional + (coupon_pay[above_coupon_mat] if ac.memory_coupon else coupon_amount)
        mat_payoff[below_put_mat] = ac.notional * (final_spot[below_put_mat] / ac.strike)
        mat_payoff[principal_only] = ac.notional
        payoffs[alive] = mat_payoff
        redemption_times[alive] = ac.maturity_years

    discount_factors = np.exp(-market.risk_free_rate * redemption_times)
    pv = payoffs * discount_factors
    price = float(np.mean(pv))
    std_err = float(np.std(pv) / np.sqrt(n_paths))
    autocall_prob = float(called.mean()) * 100
    knockin_prob = float((knocked_in & ~called).mean()) * 100
    avg_life = float(np.mean(redemption_times))

    # Payoff distribution
    payoff_buckets = {
        "capital_loss": float(np.mean(payoffs < ac.notional * 0.99)) * 100,
        "principal_only": float(np.mean((payoffs >= ac.notional * 0.99) & (payoffs <= ac.notional * 1.01))) * 100,
        "with_coupon": float(np.mean(payoffs > ac.notional * 1.01)) * 100,
    }

    return {
        "price": round(price, 2),
        "std_error": round(std_err, 2),
        "price_pct": round(price / ac.notional * 100, 2),
        "autocall_prob": round(autocall_prob, 1),
        "knockin_prob": round(knockin_prob, 1),
        "avg_life": round(avg_life, 2),
        "autocall_counts": autocall_counts,
        "obs_labels": [f"Q{k+1}" for k in range(n_obs)],
        "payoff_buckets": payoff_buckets,
    }


def vol_sensitivity(ac, market, n_paths=50_000):
    vols = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]
    prices, autocalls, knockins = [], [], []
    for v in vols:
        m = MarketParams(market.risk_free_rate, market.dividend_yield, v)
        r = price_autocallable(ac, m, n_paths=n_paths, seed=123)
        prices.append(r["price_pct"])
        autocalls.append(r["autocall_prob"])
        knockins.append(r["knockin_prob"])
    return {
        "vols": [f"{v:.0%}" for v in vols],
        "prices": prices,
        "autocalls": autocalls,
        "knockins": knockins,
    }


def spot_sensitivity(ac, market, n_paths=50_000):
    offsets = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]
    labels, prices, autocalls = [], [], []
    for pct in offsets:
        new_spot = ac.spot * (1 + pct / 100)
        ac2 = AutocallableParams(
            spot=new_spot, strike=ac.strike,
            autocall_barrier=ac.autocall_barrier, coupon_barrier=ac.coupon_barrier,
            put_barrier=ac.put_barrier, coupon_rate=ac.coupon_rate,
            notional=ac.notional, maturity_years=ac.maturity_years,
            obs_freq=ac.obs_freq, memory_coupon=ac.memory_coupon,
        )
        r = price_autocallable(ac2, market, n_paths=n_paths, seed=123)
        labels.append(f"{pct:+d}%")
        prices.append(r["price_pct"])
        autocalls.append(r["autocall_prob"])
    return {"labels": labels, "prices": prices, "autocalls": autocalls}


def generate_sample_paths(market, spot, maturity, n_paths=8, seed=99):
    total_steps = int(maturity * 252)
    paths = simulate_paths(market, spot, maturity, total_steps, n_paths, seed)
    # Downsample to ~50 points for chart
    step = max(1, total_steps // 50)
    indices = list(range(0, total_steps + 1, step))
    if indices[-1] != total_steps:
        indices.append(total_steps)
    times = [round(i / 252, 3) for i in indices]
    sampled = paths[:, indices]
    return {
        "times": times,
        "paths": [sampled[i].tolist() for i in range(n_paths)],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/price", methods=["POST"])
def api_price():
    d = request.json
    ac = AutocallableParams(
        spot=float(d.get("spot", 5700)),
        strike=float(d.get("strike", 5700)),
        autocall_barrier=float(d.get("autocall_barrier", 1.0)),
        coupon_barrier=float(d.get("coupon_barrier", 0.70)),
        put_barrier=float(d.get("put_barrier", 0.60)),
        coupon_rate=float(d.get("coupon_rate", 0.08)),
        notional=float(d.get("notional", 1000)),
        maturity_years=float(d.get("maturity_years", 3)),
        obs_freq=int(d.get("obs_freq", 4)),
        memory_coupon=bool(d.get("memory_coupon", True)),
    )
    market = MarketParams(
        risk_free_rate=float(d.get("risk_free_rate", 0.04)),
        dividend_yield=float(d.get("dividend_yield", 0.015)),
        volatility=float(d.get("volatility", 0.18)),
    )
    n_paths = int(d.get("n_paths", 100_000))

    results = price_autocallable(ac, market, n_paths=n_paths)
    vol_sens = vol_sensitivity(ac, market)
    spot_sens = spot_sensitivity(ac, market)
    sample = generate_sample_paths(market, ac.spot, ac.maturity_years)

    return jsonify({
        "results": results,
        "vol_sensitivity": vol_sens,
        "spot_sensitivity": spot_sens,
        "sample_paths": sample,
        "params": {
            "spot": ac.spot, "strike": ac.strike,
            "autocall_barrier": ac.autocall_barrier,
            "coupon_barrier": ac.coupon_barrier,
            "put_barrier": ac.put_barrier,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
