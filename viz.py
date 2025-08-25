# !pip install plotly
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def smooth(x, window=5):
    if window <= 1 or len(x) < window:
        return np.array(x)
    w = np.ones(window) / window
    return np.convolve(x, w, mode="valid")

def plot_trainer_history_plotly(trainer, output_dir="ag_news_lora",
                                smoothing_window=1, save_html=True, show=True):
    """
    Interactive Plotly visualization for HuggingFace Trainer logs.
    - Training loss (per logged step) -> train_loss.html
    - Evaluation metrics (all eval_*) -> eval_metrics.html
    """
    os.makedirs(output_dir, exist_ok=True)
    logs = trainer.state.log_history  # list of dicts

    # --- Training loss (per step) ---
    train_entries = [l for l in logs if "loss" in l and "step" in l]
    if len(train_entries) > 0:
        steps = [int(l["step"]) for l in train_entries]
        losses = [float(l["loss"]) for l in train_entries]

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=steps, y=losses, mode="lines+markers", name="loss (raw)",
            hovertemplate="step: %{x}<br>loss: %{y:.6f}<extra></extra>"
        ))

        if smoothing_window and smoothing_window > 1 and len(losses) >= smoothing_window:
            y_s = smooth(np.array(losses), smoothing_window)
            x_s = steps[(smoothing_window - 1):]  # align x for 'valid' conv
            fig_loss.add_trace(go.Scatter(
                x=x_s, y=y_s, mode="lines", name=f"smoothed (w={smoothing_window})",
                hovertemplate="step: %{x}<br>loss: %{y:.6f}<extra></extra>"
            ))

        fig_loss.update_layout(
            title="Training loss (per logged step)",
            xaxis_title="step",
            yaxis_title="loss",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if save_html:
            outp = os.path.join(output_dir, "train_loss.html")
            fig_loss.write_html(outp, include_plotlyjs="cdn")
            print(f"Saved training loss HTML -> {outp}")
        if show:
            fig_loss.show()
    else:
        print("No training-loss entries found in trainer.state.log_history.")

    # --- Evaluation metrics ---
    eval_entries = [l for l in logs if any(k.startswith("eval_") for k in l.keys())]
    if len(eval_entries) == 0:
        print("No evaluation entries found in trainer.state.log_history.")
        return

    # Collect metric names (strip "eval_" prefix)
    metric_names = set()
    for e in eval_entries:
        for k in e.keys():
            if k.startswith("eval_"):
                metric_names.add(k[len("eval_"):])

    fig_eval = go.Figure()
    any_epoch_values = any("epoch" in e for e in eval_entries)

    for m in sorted(metric_names):
        xs, ys = [], []
        for e in eval_entries:
            val = e.get(f"eval_{m}", None)
            if val is None:
                continue
            # Prefer epoch for x-axis if present; fallback to step; else create indices later
            x = e.get("epoch", e.get("step", None))
            xs.append(float(x) if x is not None else None)
            ys.append(float(val))
        if len(xs) == 0:
            continue
        # If some x are None, replace with 1..N sequence
        if any(x is None for x in xs):
            xs = list(range(1, len(ys) + 1))
        fig_eval.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=m,
            hovertemplate="step: %{x}<br>loss: %{y:.6f}<extra></extra>"
        ))

    xaxis_label = "epoch" if any_epoch_values else "eval step / index"
    fig_eval.update_layout(
        title="Evaluation metrics",
        xaxis_title=xaxis_label,
        yaxis_title="metric value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if save_html:
        outp = os.path.join(output_dir, "eval_metrics.html")
        fig_eval.write_html(outp, include_plotlyjs="cdn")
        print(f"Saved evaluation metrics HTML -> {outp}")
    if show:
        fig_eval.show()

# Example usage (call after trainer.train()):
plot_trainer_history_plotly(trainer, output_dir=args.output_dir, smoothing_window=5, save_html=True, show=True)
