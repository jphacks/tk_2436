import json
from itertools import combinations

import numpy as np

# 化粧品データベースの読み込み
with open(
    "combinatorial_optimization/cosmetics_database.json", "r", encoding="utf-8"
) as f:
    data = json.load(f)

# TODO: Web側で入力された情報を取得
# 現在の肌トーン情報
current_tone = {"色相": 40, "彩度": 20, "明度": 30}

# TODO: Web側で入力された情報を取得
# 目標の肌トーン情報
target_tone = {"色相": 90, "彩度": 80, "明度": 85}

# DPテーブルの初期化
dp = {}

# DPの初期状態（現在の肌トーン）
initial_state = tuple(current_tone.values())
dp[initial_state] = (float("inf"), None)  # (差分の最小値, 使用した組み合わせ)

# 商品タイプのリストを作成
product_types = set(item["type"] for item in data)


# 遷移を行う関数
def update_dp(selected_products):
    # 各特徴量の加算
    new_tone = {
        feature: current_tone[feature]
        + sum(product[feature] for product in selected_products)
        for feature in target_tone
    }

    # 差分の計算
    diff = sum(
        (new_tone[feature] - target_tone[feature]) ** 2 for feature in target_tone
    )

    # DPテーブル更新
    new_state = tuple(new_tone.values())
    if new_state not in dp or dp[new_state][0] > diff:
        dp[new_state] = (diff, [product["name"] for product in selected_products])


# 各商品タイプごとに商品をグループ化
grouped_products = {ptype: [] for ptype in product_types}
for product in data:
    grouped_products[product["type"]].append(product)

# 同じタイプを選ばない条件で全ての組み合わせを生成
for r in range(1, len(data) + 1):
    for product_combination in combinations(data, r):
        if len(set(p["type"] for p in product_combination)) == len(product_combination):
            update_dp(product_combination)

# 最適解の探索
best_state = min(dp, key=lambda x: dp[x][0])

# 結果の表示
if dp[best_state][1] is not None:
    print("最適な組み合わせ:")
    for product_name in dp[best_state][1]:
        print(product_name)
    print("最終的な肌のトーン:", best_state)
    print("目標との差の最小距離:", np.sqrt(dp[best_state][0]))
else:
    print("最適な組み合わせが見つかりませんでした。")
