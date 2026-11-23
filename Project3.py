import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from utils import load_or_update_tw50, build_features

# ===== 可調參數 =====
VOLUME_MIN = 500000
FORCE_PICKS = True
NUM_PICKS = 10

feature_cols = [
    'is_listed', '當日漲跌幅', '當日振幅', '5日乖離率', '20日乖離率',
    '均線差比率', '量比', '過去5日漲跌幅', '過去5日波動率',
    '20日價格位置', 'RSI_14', '大盤_20日乖離率', '大盤_5日漲跌幅'
]

print("🔍 1. 載入最新科技股與大盤數據...")
df = pd.read_csv("all_tech_stocks_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['stock_id', 'Date']).reset_index(drop=True)

tw50 = load_or_update_tw50("2025-01-01")
df = pd.merge(df, tw50[['Date', '大盤_20日乖離率', '大盤_5日漲跌幅']], on='Date', how='left')

print("⚙️ 2. 計算特徵...")
df = build_features(df)

latest_date = df['Date'].max()
today_df = df[df['Date'] == latest_date].dropna(subset=feature_cols).copy()

if today_df.empty:
    print("⚠️ 今日無有效資料，無法產生選股。")
    raise SystemExit

print("\n📊 === 診斷資訊 ===")
print(f"latest_date = {latest_date}")
print(f"today_df 筆數 = {len(today_df)}")

print("\ntoday_df 特徵統計（min / max / mean）：")
for col in feature_cols:
    print(f"  {col:12s}: min={today_df[col].min():+.4f}, max={today_df[col].max():+.4f}, mean={today_df[col].mean():+.4f}")

print("\n📦 3. 載入模型並校準信心度...")
model = joblib.load("xgb_model.pkl")
test_data = joblib.load("test_data.pkl")

calib_input = model.predict_proba(test_data['X_test'])[:, 1].reshape(-1, 1)
calibrator = LogisticRegression()
calibrator.fit(calib_input, test_data['y_test'])

raw_probs = model.predict_proba(today_df[feature_cols])[:, 1].reshape(-1, 1)
today_df['AI_Prob'] = calibrator.predict_proba(raw_probs)[:, 1]

median_prob = today_df['AI_Prob'].median()
high_ratio = (today_df['AI_Prob'] > 0.9).mean()
print(f"\n📊 校準後信心度分布：median={median_prob:.3f}, >0.9 比例={high_ratio:.2%}")
if median_prob > 0.8:
    print("⚠️ 警告：信心度中位數過高，模型可能過度自信。")
else:
    print("✅ 信心度分布正常。")

print("\ntop 5 高信心度股票：")
top5 = today_df.nlargest(5, 'AI_Prob')
print(top5[['stock_id', 'AI_Prob', '大盤_20日乖離率', '過去5日波動率', '當日振幅']].to_string(index=False))

market_bias = today_df['大盤_20日乖離率'].values[0] if len(today_df) > 0 else 0.0
print(f"\n📊 最新交易日 ({latest_date.strftime('%Y-%m-%d')}) 大盤 20 日乖離率：{market_bias:+.2%}")

if market_bias < -0.01:
    print("⚠️ 警告：目前大盤處於偏空跌勢，請謹慎操作。")

if market_bias < -0.01 and not FORCE_PICKS:
    print("⛔ 偏空環境下跳過選股。")
    picks = pd.DataFrame()
else:
    picks = today_df[
        (today_df['Volume_Size'] >= VOLUME_MIN)
    ].sort_values(by='AI_Prob', ascending=False).head(NUM_PICKS)

print(f"\n🎯 明日 AI 精選科技股關注名單（取前 {NUM_PICKS} 名，按校準後信心度排序）：")
if picks.empty:
    print("今日無符合流動性條件的標的。")
else:
    for _, row in picks.iterrows():
        vol_lots = int(row['Volume_Size'] / 1000)
        print(f"代號: {row['stock_id']} ({row['stock_name']}) | 收盤價: {row['Close_Price']:.2f} | 成交量: {vol_lots} 張 | AI信心度: {row['AI_Prob']:.2%}")