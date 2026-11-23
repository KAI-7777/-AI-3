import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from utils import load_or_update_tw50, build_features

SPLIT_DATE = pd.to_datetime("2026-08-01")
BUFFER_DATE = pd.to_datetime("2026-07-18")
RUN_CV = False
TRADING_COST = 0.004
BACKTEST_THRESHOLD = 0.60

feature_cols = [
    'is_listed', '當日漲跌幅', '當日振幅', '5日乖離率', '20日乖離率',
    '均線差比率', '量比', '過去5日漲跌幅', '過去5日波動率',
    '20日價格位置', 'RSI_14', '大盤_20日乖離率', '大盤_5日漲跌幅'
]

print("💾 1. 載入個股歷史數據...")
df = pd.read_csv("all_tech_stocks_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['stock_id', 'Date']).reset_index(drop=True)

print("🌐 2. 下載 0050 數據建立大盤環境特徵...")
tw50 = load_or_update_tw50("2025-01-01")
df = pd.merge(df, tw50[['Date', '大盤_20日乖離率', '大盤_5日漲跌幅']], on='Date', how='left')

print("⚙️ 3. 計算進階技術指標特徵...")
df = build_features(df)
df = df.dropna(subset=feature_cols + ['Real_Return']).reset_index(drop=True)

X = df[feature_cols]
y = df['Target']

train_mask = df['Date'] < BUFFER_DATE
test_mask  = df['Date'] >= SPLIT_DATE
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

if RUN_CV:
    from sklearn.model_selection import TimeSeriesSplit
    print("🔬 執行時間序列 CV...")
    tscv = TimeSeriesSplit(n_splits=3)
    for tr, te in tscv.split(X_train):
        m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, eval_metric='logloss')
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        acc = (m.predict(X_train.iloc[te]) == y_train.iloc[te]).mean()
        print(f"  fold accuracy: {acc:.2%}")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
weight_ratio = neg_count / pos_count if pos_count > 0 else 1.0

print("🚀 4. 開始訓練升級版 XGBoost 模型...")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.03,
    scale_pos_weight=weight_ratio, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=2.0, random_state=42, eval_metric='logloss'
)
model.fit(X_train, y_train)

joblib.dump(model, "xgb_model.pkl")
joblib.dump({'X_test': X_test, 'y_test': y_test}, "test_data.pkl")
print("✅ 模型與測試資料已儲存")

y_probs = model.predict_proba(X_test)[:, 1]
test_returns = df.loc[test_mask, 'Real_Return']

print("\n🎯 升級版模型：不同信心門檻下的勝率與平均報酬率：")
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    mask = (y_probs >= threshold)
    count = mask.sum()
    if count > 0:
        win_3pct = (y_test[mask] == 1).mean()
        win_pos  = (test_returns[mask] > 0).mean()
        avg_ret  = test_returns[mask].mean()
        print(f"門檻 >= {threshold:.0%} | 出手: {count:5d} 次 | 大漲>=3%率: {win_3pct:.2%} | 正報酬勝率: {win_pos:.2%} | 平均5日報酬: {avg_ret:+.2%}")
    else:
        print(f"門檻 >= {threshold:.0%} | 出手:     0 次")

print(f"\n📊 信心度分布：min={y_probs.min():.3f}, median={np.median(y_probs):.3f}, max={y_probs.max():.3f}")
print(f"   AI_Prob > 0.9 的比例：{(y_probs > 0.9).mean():.2%}")
print(f"   訓練集正樣本比例：{y_train.mean():.2%}")
print(f"   測試集正樣本比例：{y_test.mean():.2%}")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n🔥 升級版 AI 特徵重要性排名：")
print(importances)

print("\n💼 5. 開始進行模擬資金回測...")

bt_df = df.loc[test_mask, ['Date', 'stock_id', 'stock_name', 'Real_Return', '過去5日波動率', '大盤_20日乖離率']].copy()
bt_df['AI_Prob'] = y_probs

cond = (
    (bt_df['AI_Prob'] >= BACKTEST_THRESHOLD) &
    (bt_df['大盤_20日乖離率'] > -0.01)
)
filtered_signals = bt_df[cond].copy()

daily_top3 = (
    filtered_signals
    .sort_values(by=['Date', 'AI_Prob'], ascending=[True, False])
    .groupby('Date').head(3)
    .copy()
)
daily_top3['Net_Return'] = daily_top3['Real_Return'] - TRADING_COST

if len(daily_top3) > 0:
    print(f"📅 測試期間：{len(bt_df['Date'].unique())} 天（出手 {len(daily_top3['Date'].unique())} 天）")
    print(f"🎯 總買進：{len(daily_top3)} 筆 | 單筆正報酬勝率：{(daily_top3['Net_Return'] > 0).mean():.2%}")
    print(f"💵 平均每筆淨報酬：{daily_top3['Net_Return'].mean():+.2%}")
else:
    print("本期間無符合條件的交易訊號。")