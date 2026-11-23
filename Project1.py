import time
import os
import pandas as pd
from datetime import timedelta
from FinMind.data import DataLoader
import yfinance as yf

HISTORY_START = "2025-01-01"
OUTPUT_FILE = "all_tech_stocks_data.csv"
MAX_RETRIES = 3
RETRY_SLEEP = 2.0

api = DataLoader()

print("🔍 1. 開始利用 FinMind 獲取最新上市櫃股票清單...")

try:
    df_info = api.taiwan_stock_info()

    tech_categories = [
        "半導體業", "電腦及週邊設備業", "光電業", "通信網路業",
        "電子零組件業", "電子通路業", "資訊服務業", "其他電子業"
    ]

    df_tech = df_info[
        df_info["industry_category"].isin(tech_categories) &
        df_info["type"].isin(["twse", "tpex"]) &
        (df_info["stock_id"].str.len() == 4)
    ]

    stock_list = {}
    for _, row in df_tech.iterrows():
        sid = row["stock_id"]
        suffix = ".TW" if row["type"] == "twse" else ".TWO"
        stock_list[f"{sid}{suffix}"] = {
            "name": row["stock_name"],
            "is_listed": 1 if row["type"] == "twse" else 0
        }

    total_count = len(stock_list)
    print(f"🎯 成功鎖定全台股共 {total_count} 檔上市櫃科技股！")

except Exception as e:
    print(f"❌ 獲取股票清單失敗，錯誤原因: {e}")
    stock_list = {}

if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
    existing["Date"] = pd.to_datetime(existing["Date"])
    last_date = existing["Date"].max()
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    existing_ids = set(existing["stock_id"].astype(str))
    print(f"📦 偵測到既有資料，最後日期 {last_date.date()}，將從 {start_date} 增量下載。")
else:
    existing = None
    existing_ids = set()
    start_date = HISTORY_START
    print("📦 無既有資料，從起始日全量下載。")

def download_with_retry(symbol, start, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(symbol, start=start, progress=False, group_by='ticker')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f" (重試 {attempt}/{retries}: {e})", end="", flush=True)
            time.sleep(RETRY_SLEEP)
    return pd.DataFrame()

if stock_list:
    print("\n🚀 2. 開始批次下載所有科技股的歷史數據...")
    all_data = []
    success_count = 0

    if pd.to_datetime(start_date) > pd.Timestamp.today():
        print("✅ 資料已是最新，無需下載。")
    else:
        for i, (symbol, info) in enumerate(stock_list.items(), 1):
            sid = symbol.split(".")[0]
            sym_start = HISTORY_START if (existing is None or sid not in existing_ids) else start_date

            print(f"[{i}/{total_count}] 正在下載 {symbol} ({info['name']})...", end="", flush=True)

            try:
                df = download_with_retry(symbol, sym_start)

                if df.empty or len(df) < 25:
                    print(" ❌ (資料太少或無資料，跳過)")
                    continue

                df = df.reset_index()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[1] if col[1] != '' else col[0] for col in df.columns]

                if "Close" not in df.columns:
                    print(" ❌ (缺少 Close 欄位)")
                    continue

                df["Date"] = pd.to_datetime(df["Date"])

                df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0) & (df["Open"] > 0) & (df["Volume"] >= 0)]
                if len(df) < 25:
                    print(" ❌ (品質過濾後資料不足，跳過)")
                    continue

                df["MA_5"] = df["Close"].rolling(window=5).mean()
                df["MA_20"] = df["Close"].rolling(window=20).mean()

                df["stock_id"] = sid
                df["stock_name"] = info["name"]
                df["is_listed"] = info["is_listed"]

                df = df.rename(columns={
                    "Open": "Open_Price",
                    "High": "High_Price",
                    "Low": "Low_Price",
                    "Close": "Close_Price",
                    "Volume": "Volume_Size"
                })

                df = df.dropna(subset=["MA_20"])

                df = df[[
                    "Date", "stock_id", "stock_name", "is_listed",
                    "Open_Price", "High_Price", "Low_Price", "Close_Price", "Volume_Size",
                    "MA_5", "MA_20"
                ]]

                all_data.append(df)
                success_count += 1
                print(" 行 ✅")

                if i % 30 == 0:
                    time.sleep(1.5)

            except Exception as e:
                print(f" ❌ (錯誤: {e})")

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)

        if existing is not None:
            final_table = pd.concat([existing, new_df], ignore_index=True)
            final_table = final_table.drop_duplicates(subset=["stock_id", "Date"], keep="last")
        else:
            final_table = new_df

        final_table = final_table.sort_values(["stock_id", "Date"]).reset_index(drop=True)
        final_table.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n📊 全部下載完成！成功收集 {success_count} 檔股票數據。")
        print(f"💾 大數據表格已成功儲存為 '{OUTPUT_FILE}'（共 {len(final_table)} 筆）。")
    else:
        print("\n📊 本次無新增資料，既有檔案維持不變。")