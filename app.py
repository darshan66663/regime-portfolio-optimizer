
import streamlit as st
import pandas as pd, numpy as np, json
import plotly.graph_objects as go

st.set_page_config(page_title="Regime-Conditional Portfolio Optimizer",
                   page_icon="📊", layout="wide")

@st.cache_data
def load():
    return (pd.read_parquet("data/weights.parquet"),
            pd.read_parquet("data/equity_curves.parquet"),
            pd.read_parquet("data/prices.parquet"),
            pd.read_parquet("data/regimes.parquet"),
            pd.read_parquet("data/fold_results.parquet"),
            pd.read_parquet("data/latest_portfolio.parquet"),
            json.load(open("data/metrics.json")))

W, curves, prices, regimes, folds_df, latest, M = load()

st.title("Regime-Conditional Deep Portfolio Optimization")
st.caption(f"{M['n_stocks']} stocks · {M['period']} · out-of-sample walk-forward")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Performance", "💼 Portfolio", "🔬 Validation", "📊 Explorer"])

with tab1:
    c = st.columns(4)
    c[0].metric("CAGR", f"{M['model']['CAGR%']:.1f}%",
                f"{M['model']['CAGR%'] - M['benchmark']['CAGR%']:+.1f}% vs EW")
    c[1].metric("Sharpe", f"{M['model']['Sharpe']:.2f}",
                f"{M['model']['Sharpe'] - M['benchmark']['Sharpe']:+.2f} vs EW")
    c[2].metric("Max drawdown", f"{M['model']['maxDD%']:.1f}%")
    c[3].metric("Volatility", f"{M['model']['vol%']:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves.index, y=curves["model"],
                             name="DL Optimizer", line=dict(color="#264653", width=2)))
    fig.add_trace(go.Scatter(x=curves.index, y=curves["benchmark"],
                             name="Equal Weight", line=dict(color="#E76F51", width=2)))
    fig.update_layout(title="Growth of ₹1 (out-of-sample)", yaxis_type="log",
                      height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    dd = curves / curves.cummax() - 1
    fig2 = go.Figure()
    for col, colr in [("model", "#264653"), ("benchmark", "#E76F51")]:
        fig2.add_trace(go.Scatter(x=dd.index, y=dd[col]*100, name=col,
                                  fill="tozeroy", line=dict(color=colr)))
    fig2.update_layout(title="Drawdown %", height=300)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Latest recommended allocation")
    amount = st.number_input("Amount to invest (₹)", 10000, 100000000, 1000000,
                             step=50000)
    disp = latest.copy()
    disp["allocation ₹"] = (disp["weight"] * amount).round(0)
    disp["weight %"] = (disp["weight"] * 100).round(2)
    st.dataframe(disp[["ticker", "market", "weight %", "allocation ₹", "price"]],
                 use_container_width=True, hide_index=True)

    c = st.columns(2)
    c[0].plotly_chart(go.Figure(go.Pie(
        labels=disp["ticker"].head(15), values=disp["weight"].head(15), hole=0.4)
        ).update_layout(title="Top 15 positions", height=400),
        use_container_width=True)
    bym = disp.groupby("market")["weight"].sum()
    c[1].plotly_chart(go.Figure(go.Bar(x=bym.index, y=bym.values*100)
        ).update_layout(title="Allocation by market (%)", height=400),
        use_container_width=True)

with tab3:
    st.subheader("Does this actually work?")
    st.warning(
        f"**Deflated Sharpe Ratio: {M['dsr']:.1f}%** (threshold 95%). "
        f"Accounting for {M['n_trials']} tested configurations, the strategy's "
        f"Sharpe of {M['model']['Sharpe']:.2f} is not statistically "
        f"distinguishable from what chance would produce.")

    c = st.columns(3)
    c[0].metric("Mean IC", f"{M['ic_mean']:+.4f}",
                f"{M['ic_folds_positive']}/{M['ic_folds_total']} folds positive")
    c[1].metric("IC vs momentum", f"{M['ic_vs_momentum']:+.4f}",
                f"paired t = {M['ic_vs_momentum_t']}")
    c[2].metric("Skew / kurtosis", f"{M['skew']:.2f} / {M['kurtosis']:.1f}")

    fig = go.Figure(go.Bar(
        x=folds_df["fold"], y=folds_df["mean_ic"],
        marker_color=["#2A9D8F" if v > 0 else "#E76F51" for v in folds_df["mean_ic"]]))
    fig.update_layout(title="Information Coefficient by walk-forward fold", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ablation studies")
    st.dataframe(pd.DataFrame(M["ablations"]), use_container_width=True,
                 hide_index=True)

with tab4:
    tick = st.selectbox("Stock", sorted(prices.columns))
    p = prices[tick].dropna()
    fig = go.Figure(go.Scatter(x=p.index, y=p.values, line=dict(color="#264653")))
    fig.update_layout(title=f"{tick} — price (INR)", height=350)
    st.plotly_chart(fig, use_container_width=True)

    hist = W[tick]
    fig2 = go.Figure(go.Scatter(x=hist.index, y=hist.values*100,
                                fill="tozeroy", line=dict(color="#2A9D8F")))
    fig2.update_layout(title=f"{tick} — portfolio weight over time (%)", height=300)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Educational project. Not investment advice. Past performance does "
           "not predict future returns.")
