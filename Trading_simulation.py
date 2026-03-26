import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LSTM_PATH = "/Users/antonsalonen/Desktop/OMXH25_LSTM_forecasts.csv"
ARIMA111_PATH = "/Users/antonsalonen/Desktop/ARIMA_111_forecasts.csv"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.024486

# Transaction costs from 0.0% to 0.5% in 0.05%-point increments
TRANSACTION_COSTS = np.arange(0.0, 0.003, 0.0005)

#Load and align data
strategy_df = pd.read_csv(LSTM_PATH, index_col=0, parse_dates=True).rename(
    columns={"Forecast_Close": "LSTM_Forecast"}
)
arima111 = pd.read_csv(ARIMA111_PATH, index_col=0, parse_dates=True)
arima111.columns = ["ARIMA_111_Forecast"]

strategy_df = strategy_df.join(arima111, how="left")

# Previous close
strategy_df["Prev_Close"] = strategy_df["Actual_Close"].shift(1)

# Predicted returns
strategy_df["LSTM_Return_Pred"] = (
    (strategy_df["LSTM_Forecast"] - strategy_df["Prev_Close"])
    / strategy_df["Prev_Close"]
)

strategy_df["ARIMA_111_Return_Pred"] = (
    (strategy_df["ARIMA_111_Forecast"] - strategy_df["Prev_Close"])
    / strategy_df["Prev_Close"]
)

# Actual returns
strategy_df["Return_Actual"] = (
    (strategy_df["Actual_Close"] - strategy_df["Prev_Close"])
    / strategy_df["Prev_Close"]
)

# Buy-and-hold
strategy_df["BuyHold_Return"] = strategy_df["Return_Actual"]
strategy_df["BuyHold_Equity"] = (1 + strategy_df["BuyHold_Return"]).cumprod()

# Performance metrics
def performance_metrics(returns):
    mean_daily = returns.mean()
    vol_daily = returns.std()
    annual_return = (1 + mean_daily) ** TRADING_DAYS - 1
    annual_vol = vol_daily * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol
    downside = returns[returns < 0].std()
    sortino = (annual_return - RISK_FREE_RATE) / (downside * np.sqrt(TRADING_DAYS))
    return annual_return, annual_vol, sharpe, sortino

summary_rows = []
performance_rows = []
equity_results = {}


# Test strategies for each cost
for tc in TRANSACTION_COSTS:
    temp = strategy_df.copy()

    # Positions: only trade if predicted return exceeds transaction cost
    temp["LSTM_Position"] = np.where(
        temp["LSTM_Return_Pred"] > tc, 1,
        np.where(temp["LSTM_Return_Pred"] < -tc, -1, 0)
    )

    temp["ARIMA_111_Position"] = np.where(
        temp["ARIMA_111_Return_Pred"] > tc, 1,
        np.where(temp["ARIMA_111_Return_Pred"] < -tc, -1, 0)
    )

    # Gross returns
    temp["LSTM_Return"] = temp["LSTM_Position"] * temp["Return_Actual"]
    temp["ARIMA_111_Return"] = temp["ARIMA_111_Position"] * temp["Return_Actual"]

    # Net returns
    temp["LSTM_Return_Net"] = (temp["LSTM_Return"] - np.where(temp["LSTM_Position"] != 0, tc, 0))
    temp["ARIMA_111_Return_Net"] = (temp["ARIMA_111_Return"]- np.where(temp["ARIMA_111_Position"] != 0, tc, 0))

    # Equity curves
    temp["LSTM_Equity"] = (1 + temp["LSTM_Return_Net"]).cumprod()
    temp["ARIMA_111_Equity"] = (1 + temp["ARIMA_111_Return_Net"]).cumprod()
    equity_results[tc] = temp[[
        "LSTM_Equity",
        "ARIMA_111_Equity",
        "BuyHold_Equity"
    ]].copy()

    #Summary
    for model_name, pos_col, gross_ret_col, net_ret_col, eq_col in [
        ("LSTM", "LSTM_Position", "LSTM_Return", "LSTM_Return_Net", "LSTM_Equity"),
        ("ARIMA(1,1,1)", "ARIMA_111_Position", "ARIMA_111_Return", "ARIMA_111_Return_Net", "ARIMA_111_Equity"),
    ]:
        active = temp[pos_col] != 0
        n_trades = active.sum()
        hit_rate = (temp.loc[active, gross_ret_col] > 0).mean() if n_trades > 0 else np.nan
        total_return = temp[eq_col].iloc[-1] - 1

        summary_rows.append({
            "Transaction_Cost_%": tc * 100,
            "Strategy": model_name,
            "Trades": int(n_trades),
            "Hit_Rate": hit_rate,
            "Total_Return": total_return
        })

        annual_return, annual_vol, sharpe, sortino = performance_metrics(temp[net_ret_col])

        performance_rows.append({
            "Transaction_Cost_%": tc * 100,
            "Strategy": model_name,
            "Annual_Return": annual_return,
            "Annual_Volatility": annual_vol,
            "Sharpe_Ratio": sharpe,
            "Sortino_Ratio": sortino
        })

#Add buy-and-hold to performance table
bh_return, bh_vol, bh_sharpe, bh_sortino = performance_metrics(strategy_df["BuyHold_Return"])
performance_rows.append({
    "Transaction_Cost_%": np.nan,
    "Strategy": "Buy and Hold",
    "Annual_Return": bh_return,
    "Annual_Volatility": bh_vol,
    "Sharpe_Ratio": bh_sharpe,
    "Sortino_Ratio": bh_sortino
})

summary_df = pd.DataFrame(summary_rows)
performance_df = pd.DataFrame(performance_rows)

summary_display = summary_df.copy()
summary_display["Transaction_Cost_%"] = summary_display["Transaction_Cost_%"].round(2)
summary_display["Hit_Rate"] = summary_display["Hit_Rate"].round(4)
summary_display["Total_Return"] = summary_display["Total_Return"].round(4)

performance_display = performance_df.copy()
performance_display["Transaction_Cost_%"] = performance_display["Transaction_Cost_%"].round(2)
for col in ["Annual_Return", "Annual_Volatility", "Sharpe_Ratio", "Sortino_Ratio"]:
    performance_display[col] = performance_display[col].round(4)

print("\nSummary table")
print(summary_display.to_string(index=False))

print("\nPerformance table")
print(performance_display.to_string(index=False))

bh_total_return = strategy_df["BuyHold_Equity"].iloc[-1] - 1
print("\nBuy-and-hold total return")
print("Total return:", round(float(bh_total_return), 4))

# Plot 1 Cumulative return plot for each cost level
for tc in TRANSACTION_COSTS:
    eq = equity_results[tc]

    plt.figure(figsize=(12, 6))
    plt.plot(eq.index, eq["LSTM_Equity"], label=f"LSTM ({tc*100:.2f}% cost)")
    plt.plot(eq.index, eq["ARIMA_111_Equity"], label=f"ARIMA(1,1,1) ({tc*100:.2f}% cost)")
    plt.plot(eq.index, eq["BuyHold_Equity"], label="Buy and Hold")
    plt.title(f"Trading Strategies Comparison - Transaction Cost {tc*100:.2f}%")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot 2 Total return vs transaction costs
pivot_total_return = summary_df.pivot(
    index="Transaction_Cost_%",
    columns="Strategy",
    values="Total_Return"
)

plt.figure(figsize=(10, 6))
for col in pivot_total_return.columns:
    plt.plot(pivot_total_return.index, pivot_total_return[col], marker="o", label=col)

plt.title("Total Return vs Transaction Cost")
plt.xlabel("Transaction Cost (%)")
plt.ylabel("Total Return")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3 Number of trades vs transaction costs
pivot_trades = summary_df.pivot(
    index="Transaction_Cost_%",
    columns="Strategy",
    values="Trades"
)

plt.figure(figsize=(10, 6))
for col in pivot_trades.columns:
    plt.plot(pivot_trades.index, pivot_trades[col], marker="o", label=col)

plt.title("Number of Trades vs Transaction Cost")
plt.xlabel("Transaction Cost (%)")
plt.ylabel("Trades")
plt.legend()
plt.grid(True)
plt.show()