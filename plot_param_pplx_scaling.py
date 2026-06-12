import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot parameter-count vs perplexity scaling law on log-log axes, "
            "after subtracting embedding parameters."
        )
    )
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        required=True,
        help="Embedding dimension used by the model.",
    )
    parser.add_argument(
        "--vocab_size", type=int, required=True, help="Vocabulary size used by the model."
    )
    parser.add_argument(
        "--embedding_count_multiplier",
        type=int,
        default=1,
        help=(
            "How many embedding matrices to subtract. "
            "Use 1 for tied input/output embeddings, 2 for untied."
        ),
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="param_pplx_scaling_loglog.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show_fit",
        action="store_true",
        help="Add least-squares power-law fit line in log-log space.",
    )
    parser.add_argument(
        "--annotate_dim",
        action="store_true",
        help="Annotate points with their 'dim' value from the CSV.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Tuple[float, float, float, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[Tuple[float, float, float, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dim", "param", "pplx", "loss"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must contain columns: dim, param, pplx, loss")

        for idx, row in enumerate(reader, start=2):
            try:
                dim = float(row["dim"])
                param = float(row["param"])
                pplx = float(row["pplx"])
                loss = float(row["loss"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value at CSV line {idx}: {row}") from exc
            rows.append((dim, param, pplx, loss))

    if not rows:
        raise ValueError("CSV has no data rows.")
    return rows


def infer_loss_output_path(pplx_output_path: str) -> str:
    p = Path(pplx_output_path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_loss{p.suffix}"))
    return f"{pplx_output_path}_loss.png"


def infer_shifted_output_path(base_output_path: str) -> str:
    p = Path(base_output_path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_shifted{p.suffix}"))
    return f"{base_output_path}_shifted.png"


def infer_curved_output_path(base_output_path: str) -> str:
    p = Path(base_output_path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_curved{p.suffix}"))
    return f"{base_output_path}_curved.png"


def infer_early_points_output_path(base_output_path: str) -> str:
    p = Path(base_output_path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_early_points{p.suffix}"))
    return f"{base_output_path}_early_points.png"


def plot_metric(
    adjusted_params: np.ndarray,
    metric: np.ndarray,
    dims: np.ndarray,
    metric_name: str,
    output_path: str,
    show_fit: bool,
    annotate_dim: bool,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        adjusted_params,
        metric,
        marker="o",
        linestyle="None",
        color="#1f77b4",
        label="runs",
    )

    if show_fit and len(adjusted_params) >= 2:
        x = np.log(adjusted_params)
        y = np.log(metric)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = np.exp(intercept) * (adjusted_params ** slope)
        ax.loglog(
            adjusted_params,
            fitted,
            linestyle="-",
            color="#ff7f0e",
            label=f"fit: {metric_name.lower()} = {np.exp(intercept):.3g} * N^{slope:.3f}",
        )

    if annotate_dim:
        for d, x, y in zip(dims, adjusted_params, metric):
            ax.annotate(f"dim={int(d)}", (x, y), textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("Adjusted parameter count (param - embedding_params)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Parameter Count vs {metric_name} (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def fit_shifted_power_law(
    adjusted_params: np.ndarray, metric: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit L(N) = C + a * N^{-alpha} by grid-searching C and
    linear fitting log(L-C) = log(a) - alpha*log(N).
    """
    n = len(metric)
    if n < 3:
        raise ValueError("Need at least 3 points to fit shifted power law.")

    min_metric = float(np.min(metric))
    c_candidates = np.linspace(0.0, min_metric * 0.999, 2000, dtype=np.float64)
    x = np.log(adjusted_params)

    best = None  # (mse, C, intercept, slope)
    for c in c_candidates:
        shifted = metric - c
        if np.any(shifted <= 0.0):
            continue
        y = np.log(shifted)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        mse = float(np.mean((y - y_hat) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(c), float(intercept), float(slope))

    if best is None:
        raise ValueError("Could not fit shifted power law with positive L-C values.")

    _, c_best, intercept_best, slope_best = best
    a_best = float(np.exp(intercept_best))
    alpha_best = float(-slope_best)
    return c_best, a_best, alpha_best


def plot_shifted_metric(
    adjusted_params: np.ndarray,
    metric: np.ndarray,
    dims: np.ndarray,
    metric_name: str,
    output_path: str,
    annotate_dim: bool,
) -> Tuple[float, float, float]:
    c_best, a_best, alpha_best = fit_shifted_power_law(adjusted_params, metric)

    x = np.log(adjusted_params)
    y = np.log(metric - c_best)
    y_fit = np.log(a_best) - alpha_best * x

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "o", color="#1f77b4", label="runs")
    plt.plot(
        x,
        y_fit,
        "-",
        color="#ff7f0e",
        label=f"fit: L=C+aN^(-alpha), C={c_best:.4g}, a={a_best:.4g}, alpha={alpha_best:.4g}",
    )
    if annotate_dim:
        for d, xx, yy in zip(dims, x, y):
            plt.annotate(f"dim={int(d)}", (xx, yy), textcoords="offset points", xytext=(4, 4))

    plt.xlabel("log N")
    plt.ylabel(f"log({metric_name} - C)")
    plt.title(f"Shifted Power-Law Linearization: {metric_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return c_best, a_best, alpha_best


def plot_metric_curved_fit(
    adjusted_params: np.ndarray,
    metric: np.ndarray,
    dims: np.ndarray,
    metric_name: str,
    output_path: str,
    annotate_dim: bool,
):
    """Plot with two separate power-law fits: one for first 4 points, one for the rest."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort by params
    sort_idx = np.argsort(adjusted_params)
    params_sorted = adjusted_params[sort_idx]
    metric_sorted = metric[sort_idx]
    dims_sorted = dims[sort_idx]
    
    from scipy.optimize import curve_fit
    
    def power_law(x, a, b):
        return a * (x ** b)
    
    # Split into first 4 and rest
    n_early = min(4, len(params_sorted))
    early_params = params_sorted[:n_early]
    early_metric = metric_sorted[:n_early]
    early_dims = dims_sorted[:n_early]
    
    rest_params = params_sorted[n_early:]
    rest_metric = metric_sorted[n_early:]
    rest_dims = dims_sorted[n_early:]
    
    # Plot all points
    ax.loglog(
        params_sorted,
        metric_sorted,
        marker="o",
        linestyle="None",
        color="#1f77b4",
        markersize=6,
    )
    
    # Fit power law for early points
    try:
        popt_early, _ = curve_fit(power_law, early_params, early_metric, 
                                 p0=[early_metric[0], -0.5], maxfev=5000)
        a_early, b_early = popt_early
        
        # Extend red line far upward (to much smaller x values to see higher y)
        x_fit_early = np.logspace(
            np.log10(early_params.min() * 0.1),
            np.log10(params_sorted.max()),
            200
        )
        y_fit_early = power_law(x_fit_early, a_early, b_early)
        
        ax.loglog(
            x_fit_early,
            y_fit_early,
            linestyle="-",
            color="#d62728",
            linewidth=1,
            label=f"early fit: {metric_name} = {a_early:.3g} * N^{b_early:.3f}",
        )
    except Exception as e:
        print(f"Early fit failed: {e}")
    
    # Fit power law for rest of points
    if len(rest_params) >= 2:
        try:
            popt_rest, _ = curve_fit(power_law, rest_params, rest_metric, 
                                    p0=[rest_metric[0], -0.3], maxfev=5000)
            a_rest, b_rest = popt_rest
            
            # Extend orange line far horizontally on both sides
            x_fit_rest = np.logspace(
                np.log10(params_sorted.min()) * 0.5,
                np.log10(params_sorted.max()) * 2,
                200
            )
            y_fit_rest = power_law(x_fit_rest, a_rest, b_rest)
            
            ax.loglog(
                x_fit_rest,
                y_fit_rest,
                linestyle="-",
                color="#ff7f0e",
                linewidth=1,
                label=f"late fit: {metric_name} = {a_rest:.3g} * N^{b_rest:.3f}",
            )
        except Exception as e:
            print(f"Rest fit failed: {e}")

    if annotate_dim:
        for d, x, y in zip(dims_sorted, params_sorted, metric_sorted):
            # Adjust positioning for rightmost labels to keep them on grid
            xytext = (4, 4)
            if x > params_sorted.max() * 0.8:  # Right side points
                xytext = (-30, 4)
            ax.annotate(f"dim={int(d)}", (x, y), textcoords="offset points", xytext=xytext, fontsize=9)

    # Center plot on actual data points - use log-space calculations for consistent scaling
    x_min, x_max = params_sorted.min(), params_sorted.max()
    y_min, y_max = metric_sorted.min(), metric_sorted.max()
    
    # Calculate padding in log space
    log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
    log_y_min, log_y_max = np.log10(y_min), np.log10(y_max)
    
    x_range = log_x_max - log_x_min
    y_range = log_y_max - log_y_min
    
    x_pad = 0.1 * x_range
    y_pad = 0.1 * y_range
    
    ax.set_xlim(10 ** (log_x_min - x_pad), 10 ** (log_x_max + x_pad))
    ax.set_ylim(10 ** (log_y_min - y_pad), 10 ** (log_y_max + y_pad))

    ax.set_xlabel("Adjusted parameter count (param - embedding_params)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Parameter Count vs {metric_name} (dual power-law fit, log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def plot_metric_early_points(
    adjusted_params: np.ndarray,
    metric: np.ndarray,
    dims: np.ndarray,
    metric_name: str,
    output_path: str,
    annotate_dim: bool,
):
    """Plot with linear fit on early points, extended across wide range but plot centered on data."""
    # Filter to first 4 points or up to dim 128
    mask = dims <= 128
    if mask.sum() < 2:
        mask = np.arange(len(dims)) < 4
    
    early_params = adjusted_params[mask]
    early_metric = metric[mask]
    early_dims = dims[mask]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        adjusted_params,
        metric,
        marker="o",
        linestyle="None",
        color="#1f77b4",
        alpha=0.5,
        label="all runs",
    )
    
    # Highlight early points
    ax.loglog(
        early_params,
        early_metric,
        marker="o",
        linestyle="None",
        color="#d62728",
        markersize=8,
        label=f"early points (dim≤128, n={len(early_params)})",
    )

    if len(early_params) >= 2:
        x = np.log(early_params)
        y = np.log(early_metric)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Create long fit line but keep plot centered on data
        x_fit_extended = np.logspace(
            np.log10(adjusted_params.min()) * 0.5,  # Extend beyond data
            np.log10(adjusted_params.max()) * 2,
            200
        )
        y_fit_extended = intercept + slope * np.log(x_fit_extended)
        metric_fit_extended = np.exp(y_fit_extended)
        
        ax.loglog(
            x_fit_extended,
            metric_fit_extended,
            linestyle="-",
            color="#d62728",
            linewidth=2,
            label=f"linear fit: {metric_name.lower()} = {np.exp(intercept):.3g} * N^{slope:.3f}",
        )
        
        # Keep axis centered on actual data, not the extended line
        ax.set_xlim(adjusted_params.min() * 0.9, adjusted_params.max() * 1.1)
        ax.set_ylim(metric.min() * 0.8, metric.max() * 1.2)

    if annotate_dim:
        for d, x, y in zip(dims, adjusted_params, metric):
            color = "#d62728" if d <= 128 else "#1f77b4"
            ax.annotate(f"dim={int(d)}", (x, y), textcoords="offset points", xytext=(4, 4), color=color)

    ax.set_xlabel("Adjusted parameter count (param - embedding_params)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Parameter Count vs {metric_name} (early points fit, log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def main():
    args = parse_args()

    if args.embedding_dim <= 0:
        raise ValueError("--embedding_dim must be > 0")
    if args.vocab_size <= 0:
        raise ValueError("--vocab_size must be > 0")
    if args.embedding_count_multiplier <= 0:
        raise ValueError("--embedding_count_multiplier must be > 0")

    rows = load_rows(Path(args.csv))
    embedding_params = args.embedding_dim * args.vocab_size * args.embedding_count_multiplier

    dims = np.asarray([r[0] for r in rows], dtype=np.float64)
    raw_params = np.asarray([r[1] for r in rows], dtype=np.float64)
    pplx = np.asarray([r[2] for r in rows], dtype=np.float64)
    loss = np.asarray([r[3] for r in rows], dtype=np.float64)
    adjusted_params = raw_params - float(embedding_params)

    valid = (adjusted_params > 0.0) & (pplx > 0.0) & (loss > 0.0)
    if not np.any(valid):
        raise ValueError(
            "No valid rows left after subtracting embedding params and requiring pplx/loss > 0."
        )

    dropped = int((~valid).sum())
    if dropped > 0:
        print(
            f"Dropping {dropped} row(s) with non-positive adjusted params, pplx, or loss."
        )

    dims = dims[valid]
    adjusted_params = adjusted_params[valid]
    pplx = pplx[valid]
    loss = loss[valid]

    order = np.argsort(adjusted_params)
    dims = dims[order]
    adjusted_params = adjusted_params[order]
    pplx = pplx[order]
    loss = loss[order]

    pplx_output_path = args.output_plot
    loss_output_path = infer_loss_output_path(pplx_output_path)
    pplx_shifted_output_path = infer_shifted_output_path(pplx_output_path)
    loss_shifted_output_path = infer_shifted_output_path(loss_output_path)

    plot_metric(
        adjusted_params=adjusted_params,
        metric=pplx,
        dims=dims,
        metric_name="Perplexity",
        output_path=pplx_output_path,
        show_fit=args.show_fit,
        annotate_dim=args.annotate_dim,
    )
    plot_metric(
        adjusted_params=adjusted_params,
        metric=loss,
        dims=dims,
        metric_name="Loss",
        output_path=loss_output_path,
        show_fit=args.show_fit,
        annotate_dim=args.annotate_dim,
    )
    
    # Curved fit plots
    pplx_curved_output_path = infer_curved_output_path(pplx_output_path)
    loss_curved_output_path = infer_curved_output_path(loss_output_path)
    plot_metric_curved_fit(
        adjusted_params=adjusted_params,
        metric=pplx,
        dims=dims,
        metric_name="Perplexity",
        output_path=pplx_curved_output_path,
        annotate_dim=args.annotate_dim,
    )
    plot_metric_curved_fit(
        adjusted_params=adjusted_params,
        metric=loss,
        dims=dims,
        metric_name="Loss",
        output_path=loss_curved_output_path,
        annotate_dim=args.annotate_dim,
    )
    
    # Early points plots
    pplx_early_output_path = infer_early_points_output_path(pplx_output_path)
    loss_early_output_path = infer_early_points_output_path(loss_output_path)
    plot_metric_early_points(
        adjusted_params=adjusted_params,
        metric=pplx,
        dims=dims,
        metric_name="Perplexity",
        output_path=pplx_early_output_path,
        annotate_dim=args.annotate_dim,
    )
    plot_metric_early_points(
        adjusted_params=adjusted_params,
        metric=loss,
        dims=dims,
        metric_name="Loss",
        output_path=loss_early_output_path,
        annotate_dim=args.annotate_dim,
    )
    
    pplx_c, pplx_a, pplx_alpha = plot_shifted_metric(
        adjusted_params=adjusted_params,
        metric=pplx,
        dims=dims,
        metric_name="Perplexity",
        output_path=pplx_shifted_output_path,
        annotate_dim=args.annotate_dim,
    )
    loss_c, loss_a, loss_alpha = plot_shifted_metric(
        adjusted_params=adjusted_params,
        metric=loss,
        dims=dims,
        metric_name="Loss",
        output_path=loss_shifted_output_path,
        annotate_dim=args.annotate_dim,
    )

    print(f"Saved pplx plot: {pplx_output_path}")
    print(f"Saved loss plot: {loss_output_path}")
    print(f"Saved curved pplx plot: {pplx_curved_output_path}")
    print(f"Saved curved loss plot: {loss_curved_output_path}")
    print(f"Saved early pplx plot: {pplx_early_output_path}")
    print(f"Saved early loss plot: {loss_early_output_path}")
    print(f"Saved shifted pplx plot: {pplx_shifted_output_path}")
    print(f"Saved shifted loss plot: {loss_shifted_output_path}")
    print(
        f"Perplexity shifted fit: C={pplx_c:.6g}, a={pplx_a:.6g}, alpha={pplx_alpha:.6g}"
    )
    print(f"Loss shifted fit: C={loss_c:.6g}, a={loss_a:.6g}, alpha={loss_alpha:.6g}")
    print(f"Embedding params subtracted per row: {embedding_params}")


if __name__ == "__main__":
    main()
