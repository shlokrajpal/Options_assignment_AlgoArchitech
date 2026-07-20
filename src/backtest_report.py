import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from POI_mapping import POI

load_dotenv()

COLOR_BG = "#0e1117"
COLOR_GRID = "rgba(255,255,255,0.08)"

COLOR_EQUITY = "#4fc3f7"
COLOR_EQUITY_FILL = "rgba(79, 195, 247, 0.15)"
COLOR_DRAWDOWN = "#ef5350"
COLOR_DRAWDOWN_FILL = "rgba(239, 83, 80, 0.25)"

COLOR_UP = "#26a69a"
COLOR_DOWN = "#ef5350"
COLOR_FVG = "rgba(0, 229, 255, 0.15)"
COLOR_FVG_LINE = "#00e5ff"
COLOR_FIB_50 = "#ffca28"
COLOR_FIB_786 = "#ff7043"


def load_data():
    tradesheet_path = os.getenv("TRADESHEET_OUTPUT_PATH", "tradesheet.csv")
    if not os.path.exists(tradesheet_path):
        raise FileNotFoundError(f"Tradesheet not found at {tradesheet_path}. Run backtest.py first.")

    trades = pd.read_csv(tradesheet_path, parse_dates=["entry_time", "exit_time"])

    ltf_path = os.getenv("NIFTY_5min_path")
    htf_path = os.getenv("NIFTY_1hr_path")

    ltf_df = pd.read_csv(ltf_path, parse_dates=["datetime"]).set_index("datetime")
    htf_df = pd.read_csv(htf_path, parse_dates=["datetime"]).set_index("datetime")

    return trades, ltf_df, htf_df


def plot_performance_dashboard(trades: pd.DataFrame):
    """Creates a performance dashboard with Equity Curve and Rolling Max Drawdown."""
    if trades.empty:
        print("No trades available to generate a performance report.")
        return

    if "total_capital" not in trades.columns:
        starting_capital = 10_000_000.0
        trades["total_capital"] = starting_capital + trades["net_pnl"].cumsum()

    trades["equity_peak"] = trades["total_capital"].cummax()
    trades["drawdown"] = trades["total_capital"] - trades["equity_peak"]
    trades["drawdown_pct"] = (trades["drawdown"] / trades["equity_peak"]) * 100

    starting_capital = trades["total_capital"].iloc[0] - trades["net_pnl"].iloc[0]
    ending_capital = trades["total_capital"].iloc[-1]
    total_return_pct = (ending_capital / starting_capital - 1) * 100
    max_dd_idx = trades["drawdown_pct"].idxmin()
    max_dd_pct = trades.loc[max_dd_idx, "drawdown_pct"]
    max_dd_time = trades.loc[max_dd_idx, "exit_time"]

    equity_min, equity_max = trades["total_capital"].min(), trades["total_capital"].max()
    equity_pad = (equity_max - equity_min) * 0.1 or equity_max * 0.01
    equity_range = [equity_min - equity_pad, equity_max + equity_pad]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Equity Curve", "Drawdown (%)"),
    )

    fig.add_trace(go.Scatter(
        x=trades["exit_time"], y=trades["total_capital"],
        mode="lines", name="Total Capital",
        line=dict(color=COLOR_EQUITY, width=2),
        fill="tozeroy", fillcolor=COLOR_EQUITY_FILL,
        hovertemplate="%{x|%d %b %Y %H:%M}<br>Capital: ₹%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=trades["exit_time"], y=trades["drawdown_pct"],
        mode="lines", name="Drawdown (%)",
        line=dict(color=COLOR_DRAWDOWN, width=1.5),
        fill="tozeroy", fillcolor=COLOR_DRAWDOWN_FILL,
        hovertemplate="%{x|%d %b %Y %H:%M}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_annotation(
        x=max_dd_time, y=max_dd_pct, row=2, col=1,
        text=f"Max DD: {max_dd_pct:.2f}%",
        showarrow=True, arrowhead=2, arrowcolor="white",
        font=dict(color="white", size=11), bgcolor="rgba(0,0,0,0.6)",
        ax=0, ay=30,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Strategy Performance Dashboard<br>"
                f"<span style='font-size:13px;color:#9aa0a6'>"
                f"Total Return: {total_return_pct:+.2f}%  |  "
                f"Max Drawdown: {max_dd_pct:.2f}%  |  "
                f"Trades: {len(trades)}</span>"
            ),
            x=0.02, xanchor="left",
        ),
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Arial, sans-serif", size=12, color="#e8eaed"),
        height=800,
        hovermode="x unified",
        showlegend=False,
        margin=dict(t=110, l=70, r=40, b=50),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor=COLOR_GRID,
        showspikes=True, spikemode="across", spikecolor="rgba(255,255,255,0.3)",
    )
    fig.update_yaxes(title_text="Capital (₹)", tickformat=",.0f", range=equity_range,
                      showgrid=True, gridcolor=COLOR_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", showgrid=True, gridcolor=COLOR_GRID, row=2, col=1)

    output_dir = os.getenv("OUTPUT_PATH", ".")
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "performance.html"))


def _plot_single_trade(trade: pd.Series, trade_index: int, ltf_df: pd.DataFrame,
                        htf_df: pd.DataFrame, zoom_window_days: int, output_dir: str) -> None:
    entry_time, exit_time = trade["entry_time"], trade["exit_time"]

    window_start = entry_time - pd.Timedelta(days=zoom_window_days)
    window_end = exit_time + pd.Timedelta(days=zoom_window_days)
    df_slice = ltf_df[(ltf_df.index >= window_start) & (ltf_df.index <= window_end)]

    times = df_slice.index.time
    df_slice = df_slice[(times >= pd.Timestamp("09:15").time()) & (times <= pd.Timestamp("15:30").time())]

    if df_slice.empty:
        print(f"Trade #{trade_index}: no LTF data in the requested window; skipped.")
        return

    _, _, direction, fib_50, fib_786, merged_poi = POI(htf_df, entry_time)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_slice.index, open=df_slice["open"], high=df_slice["high"],
        low=df_slice["low"], close=df_slice["close"],
        increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
        increasing_fillcolor=COLOR_UP, decreasing_fillcolor=COLOR_DOWN,
        name="Nifty Spot", showlegend=False,
    ))

    zone_x0 = max(window_start, entry_time - pd.Timedelta(hours=12))
    zone_x1 = min(window_end, exit_time + pd.Timedelta(hours=6))

    if merged_poi and len(merged_poi) == 2:
        fig.add_shape(
            type="rect", x0=zone_x0, x1=zone_x1, y0=merged_poi[0], y1=merged_poi[1],
            fillcolor=COLOR_FVG, line=dict(color=COLOR_FVG_LINE, width=1),
            layer="below",
        )
        fig.add_annotation(
            x=zone_x0, y=merged_poi[1], text="FVG Zone", showarrow=False,
            font=dict(color=COLOR_FVG_LINE, size=11), xanchor="left", yanchor="bottom",
        )

    for level, color, label in [(fib_50, COLOR_FIB_50, "Fib 0.5"), (fib_786, COLOR_FIB_786, "Fib 0.786")]:
        if level is not None:
            fig.add_shape(
                type="line", x0=zone_x0, x1=zone_x1, y0=level, y1=level,
                line=dict(color=color, width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=zone_x1, y=level, text=label, showarrow=False,
                font=dict(color=color, size=11), xanchor="right", yanchor="bottom",
            )

    entry_color = COLOR_UP if trade["direction"] == "uptrend" else COLOR_DOWN
    entry_symbol = "triangle-up" if trade["direction"] == "uptrend" else "triangle-down"
    fig.add_trace(go.Scatter(
        x=[entry_time], y=[trade["entry_price"]], mode="markers+text",
        name="Entry", text=["Entry"], textposition="bottom center",
        marker=dict(symbol=entry_symbol, size=16, color=entry_color, line=dict(width=2, color="white")),
        hovertemplate=f"Entry ({trade['direction']})<br>₹%{{y:,.2f}}<extra></extra>",
    ))

    exit_color = COLOR_UP if trade["net_pnl"] > 0 else COLOR_DOWN
    fig.add_trace(go.Scatter(
        x=[exit_time], y=[trade["exit_price"]], mode="markers+text",
        name="Exit", text=[trade["exit_reason"]], textposition="top center",
        marker=dict(symbol="x", size=13, color=exit_color, line=dict(width=2, color="white")),
        hovertemplate=(
            f"Exit ({trade['exit_reason']})<br>₹%{{y:,.2f}}<br>"
            f"Net P&L: ₹{trade['net_pnl']:,.0f}<extra></extra>"
        ),
    ))

    pnl_color = COLOR_UP if trade["net_pnl"] > 0 else COLOR_DOWN
    fig.update_layout(
        title=dict(
            text=(
                f"Trade #{trade_index} — {trade['direction'].title()} "
                f"({entry_time:%d %b %Y %H:%M})<br>"
                f"<span style='font-size:13px;color:{pnl_color}'>"
                f"Net P&L: ₹{trade['net_pnl']:,.0f} ({trade['exit_reason']})</span>"
            ),
            x=0.02, xanchor="left",
        ),
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Arial, sans-serif", size=12, color="#e8eaed"),
        yaxis_title="Nifty Spot Price",
        xaxis_title="Time",
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(t=110, l=70, r=40, b=50),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=COLOR_GRID,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),           # hide weekends
            dict(bounds=[15.5, 9.25], pattern="hour"),  # hide 15:30-09:15 overnight gap
        ],
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"signal_visualization_trade_{trade_index}.html")
    fig.write_html(out_path, config={"scrollZoom": True, "displayModeBar": True})
    print(f"Trade #{trade_index} signal chart saved to '{out_path}'")


def plot_signal_visualization(trades: pd.DataFrame, ltf_df: pd.DataFrame, htf_df: pd.DataFrame,
                               zoom_window_days: int = 3, output_dir: str = None):
    if trades.empty:
        print("No trades available to visualize.")
        return

    if output_dir is None:
        output_dir = os.getenv("OUTPUT_PATH", ".")

    for trade_index, trade in trades.iterrows():
        _plot_single_trade(trade, trade_index, ltf_df, htf_df, zoom_window_days, output_dir)


if __name__ == "__main__":
    print("Generating Backtest Reports...")
    try:
        trades_df, ltf_spot, htf_spot = load_data()
        plot_performance_dashboard(trades_df)
        plot_signal_visualization(trades_df, ltf_spot, htf_spot)
        print("Done. Open the generated HTML files in your browser.")
    except Exception as e:
        print(f"Error generating reports: {e}")