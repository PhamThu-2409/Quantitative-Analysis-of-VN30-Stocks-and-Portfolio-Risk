# IMPORT THƯ VIỆN CẦN THIẾT
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import time

# time series & stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

    
# BƯỚC 1 — LẤY DANH SÁCH VN30
vn30_stock = ("ACB", "BCM", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG",
              "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
               "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE")

print("Danh sách VN30:", vn30_stock)
print("Tổng số mã:", len(vn30_stock))

start_date = "01/01/2020"
end_date = "21/11/2025"


# BƯỚC 2 — HÀM LẤY DỮ LIỆU TỪ CafeF 
def fetch_cafef_price(ticker, start_date, end_date, page_index=1, page_size=10000, save_raw=True): 
    """
    Lấy dữ liệu giá CafeF.
    start_date, end_date dạng DD/MM/YYYY.
    """
    url = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    
    params = {
        "Symbol": ticker,
        "StartDate": start_date,
        "EndDate": end_date,
        "PageIndex": page_index,
        "PageSize": page_size
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://s.cafef.vn/"
    }

    resp = requests.get(url, params=params, headers=headers, timeout=20)

    try:
        data = resp.json()
    except:
        print(f"[ERROR] JSON parse lỗi cho {ticker}")
        return pd.DataFrame()

    # Lấy danh sách row
    rows = []
    if "Data" in data and isinstance(data["Data"], dict) and "Data" in data["Data"]:
        rows = data["Data"]["Data"]
    else:
        print(f"[WARN] Không tìm thấy Data.Data cho {ticker}. Kiểm tra JSON raw.")
        return pd.DataFrame()

    # Chuyển sang DataFrame
    df = pd.DataFrame(rows)
    if not df.empty:
        df["symbol"] = ticker

    return df

all_data = []

for ticker in vn30_stock:
    print(f"Đang tải {ticker}...")
    df = fetch_cafef_price(ticker, start_date, end_date, page_index=1, page_size=10000, save_raw=True)
    
    # Nếu API thất bại cho bất kỳ ticker nào → dừng
    if df is None or df.empty:
        print("API thất bại → dùng file backup!") 
        backup_file = "VN30_raw_backup_2020_2025.csv"
        if os.path.exists(backup_file):
            df = pd.read_csv(backup_file)
            all_data.append(df)
        else:
            print(" Không có file backup → DỪNG LẠI!")
        break

    if df is not None and not df.empty: 
        all_data.append(df)
        print(f"✓ Thành công ({len(df)} dòng)")
        
        # Nghỉ 0.5s để tránh bị chặn
        time.sleep(0.5)
        
# Gộp toàn bộ
df_all = pd.concat(all_data, ignore_index=True)

# Chuẩn hóa cột date để sắp xếp
# CafeF trả date dạng string, convert ngày tháng
df_all["Ngay"] = pd.to_datetime(df_all["Ngay"], dayfirst=True, errors="coerce")

# Sắp xếp theo symbol rồi date
df_all = df_all.sort_values(["symbol", "Ngay"]).reset_index(drop=True)

df_all.to_csv("VN30_raw_2020_2025.csv", index=False, encoding="utf-8-sig")
print("Đã lưu file raw data: VN30_raw_2020_2025.csv")
print(df_all.head)

# Lấy dữ liệu VN-Index:
ticker = "VNINDEX"
print(f"Đang tải {ticker}...")

# Fetch từ CafeF
vnindex_df = fetch_cafef_price(
    ticker,
    start_date,
    end_date,
    page_index=1,
    page_size=10000,
    save_raw=True
)

if vnindex_df is not None and not vnindex_df.empty:
    all_data.append(vnindex_df)
    print(f"✓ Thành công ({len(vnindex_df)} dòng)")
    
    # Chuẩn hóa cột date để sắp xếp
    # CafeF trả date dạng string, convert ngày tháng
    vnindex_df["Ngay"] = pd.to_datetime(vnindex_df["Ngay"], dayfirst=True, errors="coerce")

    # Sắp xếp theo symbol rồi date
    vnindex_df = vnindex_df.sort_values(["Ngay"]).reset_index(drop=True)

    vnindex_df.to_csv("VNINDEX_raw_2020_2025.csv", index=False, encoding="utf-8-sig")
    print("Đã lưu file raw data: VNINDEX_raw_2020_2025.csv")
else:
    print("✗ Lỗi: Không thể tải dữ liệu VNINDEX từ API. Dùng file backup...")
    vnindex_backup = "VNINDEX_raw_backup_2020_2025.csv"
    if os.path.exists(vnindex_backup):
        vnindex_df = pd.read_csv(vnindex_backup)

# Nghỉ 0.5 giây tránh bị chặn
time.sleep(0.5)

# BƯỚC 3 — XỬ LÝ DỮ LIỆU THÔ
# 1. XỬ LÝ DỮ LIỆU CHO VN30
def preprocess_raw_data(df_all):
    """
    Tiền xử lý DataFrame thô từ CafeF:
    - Giữ các cột cần thiết: Date, Close, Adj_Close, Open, High, Low, Volume, symbol
    - Rename cột sang chuẩn
    - Convert số và ngày tháng
    """
    df = df_all.copy()
    # Chuẩn hóa tên cột 
    rename_map = {
        "Ngay": "date",
        "GiaDongCua": "close",
        "GiaDieuChinh": "adj_close",
        "GiaMoCua": "open",
        "GiaCaoNhat": "high",
        "GiaThapNhat": "low",
        'KhoiLuongKhopLenh': 'volume',
        "symbol": "ticker"
        }
    
    df.rename(columns=rename_map, inplace=True)

    # Chỉ giữ các cột cần thiết nếu có
    keep_cols = ["date", "close", "adj_close", "open", "high", "low", "volume", "ticker"]
    df = df[[c for c in keep_cols if c in df.columns]]
    
    # Chuyển symbol ngay sau date
    cols = df.columns.tolist()
    if "ticker" in cols:
        cols.remove("ticker")
        cols.insert(1, "ticker")
        df = df[cols]

    # Convert ngày tháng
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Convert numeric các cột số
    num_cols = ["close", "adj_close", "open", "high", "low", "volume", "value"]
    for c in num_cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                      .str.replace(",", "", regex=False)
            )
            
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Xử lý missing
    missing_before = df.isna().sum().sum()

    if missing_before > 0:
        print(f"[INFO] Missing detected: {missing_before} values → applying cleaning")
        
        # Loại bỏ dòng không có ngày
        df = df.dropna(subset=['date'])

        # Forward fill theo từng mã 
        df = df.groupby("ticker", group_keys=False).apply(lambda x: x.ffill())

        # Drop lại phần còn NA (đầu chuỗi)
        df = df.dropna()

        missing_after = df.isna().sum().sum()
        print(f"[INFO] Missing after cleaning: {missing_after}")
    
    return df

df_clean = preprocess_raw_data(df_all)

df_clean.to_csv("VN30_clean_2020_2025.csv", index=False, encoding="utf-8-sig")
print("Đã lưu file raw data: VN30_clean_2020_2025.csv")

# Kiểm tra 5 dòng đầu
print("Đã xử lý dữ liệu VN30")
print(df_clean.head())

# 2. XỬ LÝ DỮ LIỆU CHO VNINDEX
def preprocess_vnindex_data(vnindex_df):
    """
    Tiền xử lý DataFrame thô từ CafeF:
    - Giữ các cột cần thiết: Date, Close, Adj_Close, Open, High, Low, Volume, symbol
    - Rename cột sang chuẩn
    - Convert số và ngày tháng
    """
    df_vnindex = vnindex_df.copy()
    # Chuẩn hóa tên cột 
    rename_map = {
        "Ngay": "date",
        "GiaDongCua": "close",
        "GiaDieuChinh": "adj_close",
        "GiaMoCua": "open",
        "GiaCaoNhat": "high",
        "GiaThapNhat": "low",
        'KhoiLuongKhopLenh': 'volume',
        "symbol": "ticker"
        }
    
    df_vnindex.rename(columns=rename_map, inplace=True)

    # Chỉ giữ các cột cần thiết nếu có
    keep_cols = ["date", "close", "adj_close", "open", "high", "low", "volume", "ticker"]
    df_vnindex = df_vnindex[[c for c in keep_cols if c in df_vnindex.columns]]
    
    
    # Chuyển symbol ngay sau date
    cols = df_vnindex.columns.tolist()
    if "ticker" in cols:
        cols.remove("ticker")
        cols.insert(1, "ticker")
        df_vnindex = df_vnindex[cols]

    # Convert ngày tháng
    if "date" in df_vnindex.columns:
        df_vnindex["date"] = pd.to_datetime(df_vnindex["date"], dayfirst=True, errors="coerce")

    # Convert numeric các cột số
    num_cols = ["close", "adj_close", "open", "high", "low", "volume", "value"]
    for c in num_cols:
        if c in df_vnindex.columns:
            df_vnindex[c] = (
                df_vnindex[c].astype(str)
                      .str.replace(",", "", regex=False)
            )
            df_vnindex[c] = pd.to_numeric(df_vnindex[c], errors="coerce")

    # Xử lý missing
    missing_before = df_vnindex.isna().sum().sum()

    if missing_before > 0:
        print(f"[INFO] Missing detected: {missing_before} values → applying cleaning")
        
        # Loại bỏ dòng không có ngày
        df_vnindex = df_vnindex.dropna(subset=['date'])

        # Forward fill 
        df_vnindex = df_vnindex.ffill()

        # Drop lại phần còn NA (đầu chuỗi)
        df_vnindex = df_vnindex.dropna()

        missing_after = df_vnindex.isna().sum().sum()
        print(f"[INFO] Missing after cleaning: {missing_after}")
        
    return df_vnindex

vnindex_clean = preprocess_vnindex_data(vnindex_df)

vnindex_clean.to_csv("VNINDEX_clean_2020_2025.csv", index=False, encoding="utf-8-sig")
print("Đã lưu file raw data: VNINDEX_clean_2020_2025.csv")

# Kiểm tra 5 dòng đầu
print("Đã xử lý dữ liệu VNINDEX")
print(vnindex_clean.head())

 # 2. XỬ LÝ DỮ LIỆU CHO RISK FREE RATE
# --- 0. Đọc và chuẩn hóa dữ liệu Risk-Free Rate ---
rf_file = "Risk_Free_Rate_2020-2025.csv"
rf_df = pd.read_csv(rf_file)

def preprocess_rf_data(rf_df):
    df_rf = rf_df.copy()
    
    # Chuẩn hóa tên cột
    rename_map = {
        "Ngày": "date",
        "Lần cuối": "rate"
    }
    df_rf.rename(columns=rename_map, inplace=True)
    
    # Giữ các cột cần thiết
    df_rf = df_rf[[c for c in ["date","rate"] if c in df_rf.columns]]
    
    # Convert ngày tháng
    df_rf["date"] = pd.to_datetime(df_rf["date"], dayfirst=True, errors="coerce")
    
    # Convert rate sang decimal
    df_rf["rate"] = df_rf["rate"].astype(str).str.replace('%','').str.replace(',','').astype(float)/100
    
    # Sắp xếp theo ngày tăng dần
    df_rf = df_rf.sort_values('date').reset_index(drop=True)
    
    # Xử lý missing
    df_rf = df_rf.dropna(subset=['date', 'rate']).ffill().dropna()
    
    return df_rf

rf_clean = preprocess_rf_data(rf_df)

# Lưu file chuẩn hóa
rf_clean.to_csv("Risk_Free_Rate_clean_2020-2025.csv", index=False, encoding="utf-8-sig")
print("Đã lưu Risk-Free Rate chuẩn hóa: Risk_Free_Rate_clean_2020-2025.csv")
print(rf_clean.head())


