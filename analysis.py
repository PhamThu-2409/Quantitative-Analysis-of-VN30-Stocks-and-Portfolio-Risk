import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# --- CẤU HÌNH OUTPUT ---
os.makedirs("output", exist_ok=True)

# --- 1. ĐỌC DỮ LIỆU ---

df_vn30 = pd.read_csv("VN30_clean_2020_2025.csv", parse_dates=["date"])
df_vnindex = pd.read_csv("VNINDEX_clean_2020_2025.csv", parse_dates=["date"])

print("VN30 shape:", df_vn30.shape)
print("VNINDEX shape:", df_vnindex.shape)

# --- 2. PIVOT GIÁ ---


pivot_close_old = df_vn30.pivot(index="date", columns="ticker", values="close").sort_index()
vnindex_close = df_vnindex.set_index("date")["close"].sort_index().to_frame("VNINDEX")

# earliest date per ticker (dựa trên pivot_close)
first_dates = pivot_close_old.apply(lambda col: col.first_valid_index())
first_dates = first_dates.sort_values()
print("Ngày xuất hiện sớm nhất từng mã:\n", first_dates)

required_start = pd.to_datetime("2020-01-02")
insufficient = first_dates[first_dates > required_start].index.tolist()
sufficient = first_dates[first_dates <= required_start].index.tolist()

print("Mã thiếu dữ liệu (bắt đầu sau 2020-01-02):", insufficient)
print("Số mã đủ 5 năm:", len(sufficient))

pivot_close = pivot_close_old[sufficient].copy()
returns_monthly_full5 = pivot_close.resample("ME").last().pct_change().dropna()
# dùng returns_monthly_full5 cho CAPM/cluster/danh mục


# Kiểm tra missing (điểm EDA)
missing_stats = pivot_close.isna().sum().sort_values(ascending=False)
print("\nMissing value top 10:")
print(missing_stats.head(10))
missing_stats.to_csv("output/missing_value_report.csv")

# --- 3. TÍNH LỢI SUẤT ---

returns_daily = pivot_close.pct_change(fill_method=None).dropna()
returns_monthly = pivot_close.resample("ME").last().pct_change(fill_method=None).dropna()

market_returns_daily = vnindex_close.pct_change(fill_method=None).dropna()
market_returns_monthly = vnindex_close.resample("ME").last().pct_change(fill_method=None).dropna()

print("\nDaily returns (sample):")
print(returns_daily.head())

# Kiểm tra outliers (EDA)
zscore = (returns_daily - returns_daily.mean()) / returns_daily.std()
outliers = (np.abs(zscore) > 4).sum().sort_values(ascending=False)
outliers.to_csv("output/outlier_report_daily.csv")

# --- 4. THỐNG KÊ MÔ TẢ ---

summary_price = pivot_close.describe().T
summary_return = returns_daily.describe().T

summary_price.to_csv("output/summary_price.csv", encoding="utf-8-sig")
summary_return.to_csv("output/summary_daily_return.csv", encoding="utf-8-sig")
print("Đã lưu summary_price.csv và summary_daily_return.csv")

# --- 5. HEATMAP TƯƠNG QUAN ---

plt.figure(figsize=(14,12))
sns.heatmap(pivot_close.corr(), cmap="coolwarm", linewidths=0.5)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Correlation Heatmap VN30", fontsize=16)
plt.tight_layout()
plt.savefig("output/VN30_correlation_heatmap.png", dpi=300)
plt.close()
print("Đã lưu VN30_correlation_heatmap.png")

# --- 6. BIỂU ĐỒ PRICE NORMALIZED ---

normalized = pivot_close / pivot_close.iloc[0]
plt.figure(figsize=(14,7))
for col in normalized.columns:
    plt.plot(normalized.index, normalized[col], linewidth=0.7, alpha=0.5)
plt.title("Normalized Price Trend (VN30)", fontsize=16)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/VN30_normalized_trend.png", dpi=300)
plt.close()
print("Đã lưu VN30_normalized_trend.png")


# --- 7. ARIMA DỰ BÁO MÃ VJC  ---

# Lấy dữ liệu giá đóng cửa
vjc_price = pivot_close["VJC"].dropna().sort_index()

# Tính daily return
vjc_return = vjc_price.pct_change().dropna()

# Chuyển sang tần suất Business-Day
vjc_return = vjc_return.asfreq("B").ffill()

# ----------------- KIỂM ĐỊNH DỪNG (ADF) -----------------
adf_result = adfuller(vjc_return)
print("\nADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

with open("output/ADF_test_VJC.txt", "w") as f:
    f.write(f"ADF statistic: {adf_result[0]}\n")
    f.write(f"p-value: {adf_result[1]}\n")

# ----------------- AUTO ARIMA RETURN -----------------
arima_model = auto_arima(
    vjc_return,
    seasonal=False,
    trace=True,
    max_p=5,
    max_q=5,
    max_d=2,
    error_action="ignore",
    suppress_warnings=True
)

print("Best ARIMA model:\n", arima_model.summary())

# Fit model
model = ARIMA(vjc_return, order=arima_model.order)
model_fit = model.fit()

# ----------------- DỰ BÁO RETURN 30 NGÀY -----------------
fc_ret = model_fit.get_forecast(steps=30)
fc_df = fc_ret.summary_frame()

# Chuẩn hóa index forecast (theo ngày BD)
forecast_index = pd.date_range(
    start=vjc_return.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq="B"
)
fc_df.index = forecast_index

# ----------------- DỰ BÁO GIÁ TỪ RETURN -----------------
last_price = vjc_price.iloc[-1]
fc_price = last_price * (1 + fc_df["mean"]).cumprod()
fc_price.index = forecast_index  # đảm bảo trùng index

# ----------------- LƯU FILE -----------------
out = pd.DataFrame({
    "forecast_return": fc_df["mean"],
    "lower_ci": fc_df["mean_ci_lower"],
    "upper_ci": fc_df["mean_ci_upper"],
    "price_forecast": fc_price
})
out.to_csv("output/VJC_ARIMA_forecast_return.csv", encoding="utf-8-sig")

print("Đã lưu VJC_ARIMA_forecast_return.csv")


# ----------------- VẼ BIỂU ĐỒ LỢI SUẤT -----------------
plt.figure(figsize=(12,6))
plt.plot(vjc_return, label="Actual Return")
plt.plot(fc_df["mean"], label="Forecast Return", color="red")

plt.fill_between(
    fc_df.index,
    fc_df["mean_ci_lower"],
    fc_df["mean_ci_upper"],
    color='pink',
    alpha=0.3
)

plt.title("ARIMA Forecast – VJC Daily Return (30 Business Days Ahead)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("output/VJC_ARIMA_return_forecast.png", dpi=300)
plt.close()
print("Đã lưu VJC_ARIMA_return_forecast.png")


# ----------------- VẼ BIỂU ĐỒ GIÁ -----------------
plt.figure(figsize=(12,6))
plt.plot(vjc_price, label="Actual Price")
plt.plot(fc_price, label="Forecast Price", color="red")

plt.title("ARIMA Forecast – VJC Price (30 Business Days Ahead)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("output/VJC_ARIMA_price_forecast.png", dpi=300)
plt.close()

print("Đã lưu VJC_ARIMA_price_forecast.png")

# --- 8. CAPM HỒI QUY BETA CHUẨN ---

# Đọc risk-free
rf_df = pd.read_csv("Risk_Free_Rate_clean_2020-2025.csv", parse_dates=["date"])
rf_df = rf_df.sort_values("date").set_index("date")

# Annual → Monthly rate
rf_df["rf_monthly"] = (1 + rf_df["rate"])**(1/12) - 1

# Đồng bộ mốc thời gian
rf_sync = rf_df["rf_monthly"].resample("ME").last()
common_dates = returns_monthly.index.intersection(market_returns_monthly.index).intersection(rf_sync.index)

ret_m = returns_monthly.loc[common_dates]
mkt_m = market_returns_monthly.loc[common_dates]
rf_m = rf_sync.loc[common_dates]

# Excess return
excess_stock = ret_m.subtract(rf_m, axis=0)
excess_mkt = (mkt_m["VNINDEX"] - rf_m).rename("MKT")

# --- Hồi quy CAPM ---
capm_rows = []

for ticker in excess_stock.columns:
    df_capm = pd.concat([excess_stock[ticker], excess_mkt], axis=1).dropna()
    X = sm.add_constant(df_capm["MKT"])
    y = df_capm[ticker]
    model = sm.OLS(y, X).fit()

    capm_rows.append({
        "Ticker": ticker,
        "Alpha": model.params["const"],
        "Beta": model.params["MKT"],
        "Alpha_tstat": model.tvalues["const"],
        "Beta_tstat": model.tvalues["MKT"],
        "Alpha_pvalue": model.pvalues["const"],
        "Beta_pvalue": model.pvalues["MKT"],
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "N_obs": int(model.nobs)
    })

capm_df = pd.DataFrame(capm_rows)
capm_df.to_csv("output/CAPM_results_realRF.csv", index=False, encoding="utf-8-sig")

print("\nĐã lưu CAPM_results_realRF.csv")
print(capm_df.head())


# --- 9. DANH MỤC THEO BETA & ĐÁNH GIÁ HIỆU QUẢ (CHỈ 2 DANH MỤC THEO ĐỀ BÀI) ---

# Chia nhóm theo Beta
high_beta = capm_df[capm_df["Beta"] > 1]["Ticker"].tolist()      # Aggressive
low_beta  = capm_df[capm_df["Beta"] <= 1]["Ticker"].tolist()     # Stable

print("Aggressive (High β):", high_beta)
print("Stable (Low β):", low_beta)

# --- Hàm tính metrics ---
def portfolio_metrics_weighted(tickers, returns_df, rf=0):
    """
    Tính metrics cho danh mục:
    - Cumulative return (1/N weighted)
    - Annualized mean return
    - Annualized volatility
    - Sharpe ratio
    - Max drawdown
    """
    data = returns_df[tickers].dropna()
    weights = np.ones(len(tickers)) / len(tickers)  # 1/N weight
    port_return = (data * weights).sum(axis=1)

    # Annualized metrics
    mean_ret = port_return.mean() * 12
    vol = port_return.std() * np.sqrt(12)
    sharpe = (mean_ret - rf) / vol
    cum_ret = (1 + port_return).cumprod()

    # Max drawdown
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    max_dd = drawdown.min()

    return mean_ret, vol, sharpe, cum_ret, max_dd


# Tính metrics cho hai danh mục
metrics = {}
portfolio_groups = {
    "Aggressive (High β)": high_beta,
    "Stable (Low β)": low_beta
}

for name, group in portfolio_groups.items():
    if len(group) == 0:
        continue
    mean_ret, vol, sharpe, cum_ret, max_dd = portfolio_metrics_weighted(
        group, 
        ret_m, 
        rf=rf_m.mean()
    )
    metrics[name] = {
        "Annualized Return": mean_ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Cumulative Return": cum_ret,
        "Max Drawdown": max_dd
    }


# --- Lưu summary metrics ---
metrics_summary = pd.DataFrame([
    {
        "Portfolio": k,
        "Annualized Return": v["Annualized Return"],
        "Volatility": v["Volatility"],
        "Sharpe Ratio": v["Sharpe Ratio"],
        "Max Drawdown": v["Max Drawdown"]
    }
    for k, v in metrics.items()
])

metrics_summary.to_csv("output/Portfolio_metrics_complete.csv", index=False, encoding="utf-8-sig")
print("Đã lưu Portfolio_metrics_complete.csv")
print(metrics_summary)

# --- Vẽ cumulative return chart ---
plt.figure(figsize=(12,6))
for name, v in metrics.items():
    plt.plot(v["Cumulative Return"], label=name)
plt.plot((1 + mkt_m["VNINDEX"]).cumprod(), label="VNINDEX", color="black", linestyle="--")
plt.title("Cumulative Return: Stable vs Aggressive Portfolio", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/Portfolio_cumulative_return_complete.png", dpi=300)
plt.close()
print("Đã lưu Portfolio_cumulative_return_complete.png")


