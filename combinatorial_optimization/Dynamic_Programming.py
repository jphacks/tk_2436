import numpy as np

# 化粧品データベース
data = [
    {
        "type": "下地",
        "name": "下地 商品1",
        "輝度": 10,
        "色み": 5,
        "彩度": 15,
        "血色感": 20,
    },
    {
        "type": "下地",
        "name": "下地 商品2",
        "輝度": 20,
        "色み": 15,
        "彩度": 20,
        "血色感": 5,
    },
    {
        "type": "下地",
        "name": "下地 商品3",
        "輝度": 10,
        "色み": 20,
        "彩度": 10,
        "血色感": 10,
    },
    {
        "type": "下地",
        "name": "下地 商品4",
        "輝度": 15,
        "色み": 20,
        "彩度": 5,
        "血色感": 20,
    },
    {
        "type": "ファンデーション",
        "name": "ファンデーション 商品1",
        "輝度": 15,
        "色み": 5,
        "彩度": 5,
        "血色感": 10,
    },
    {
        "type": "ファンデーション",
        "name": "ファンデーション 商品2",
        "輝度": 10,
        "色み": 15,
        "彩度": 15,
        "血色感": 5,
    },
    {
        "type": "ファンデーション",
        "name": "ファンデーション 商品3",
        "輝度": 20,
        "色み": 15,
        "彩度": 5,
        "血色感": 20,
    },
    {
        "type": "ファンデーション",
        "name": "ファンデーション 商品4",
        "輝度": 5,
        "色み": 10,
        "彩度": 20,
        "血色感": 5,
    },
    {
        "type": "コンシーラー",
        "name": "コンシーラー 商品1",
        "輝度": 15,
        "色み": 5,
        "彩度": 5,
        "血色感": 10,
    },
    {
        "type": "コンシーラー",
        "name": "コンシーラー 商品2",
        "輝度": 10,
        "色み": 15,
        "彩度": 15,
        "血色感": 5,
    },
    {
        "type": "コンシーラー",
        "name": "コンシーラー 商品3",
        "輝度": 20,
        "色み": 15,
        "彩度": 5,
        "血色感": 20,
    },
    {
        "type": "コンシーラー",
        "name": "コンシーラー 商品4",
        "輝度": 5,
        "色み": 10,
        "彩度": 20,
        "血色感": 5,
    },
]

# 現在の肌トーン情報
current_tone = {"輝度": 40, "色み": 50, "彩度": 20, "血色感": 30}

# 目標の肌トーン情報
target_tone = {"輝度": 60, "色み": 60, "彩度": 40, "血色感": 85}

# DPテーブルの初期化
dp = {}

# DPの初期状態（現在の肌トーン）
initial_state = tuple(current_tone.values())
dp[initial_state] = (float("inf"), None)  # (差分の最小値, 使用した組み合わせ)


"""
dataを変更した場合、特徴量情報を追加する必要あり
"""
# 遷移
for i in range(len(data)):
    for j in range(i + 1, len(data)):
        for k in range(j + 1, len(data)):
            # 同じタイプを選ばない条件
            if data[i]["type"] != data[j]["type"]:
                # 各特徴量の加算
                new_tone = {
                    "輝度": current_tone["輝度"]
                    + data[i]["輝度"]
                    + data[j]["輝度"]
                    + data[k]["輝度"],
                    "色み": current_tone["色み"]
                    + data[i]["色み"]
                    + data[j]["色み"]
                    + data[k]["色み"],
                    "彩度": current_tone["彩度"]
                    + data[i]["彩度"]
                    + data[j]["彩度"]
                    + data[k]["彩度"],
                    "血色感": current_tone["血色感"]
                    + data[i]["血色感"]
                    + data[j]["血色感"]
                    + data[k]["血色感"],
                }

                # 差分の計算
                diff = sum(
                    (new_tone[feature] - target_tone[feature]) ** 2
                    for feature in target_tone
                )

                # DPテーブル更新
                new_state = tuple(new_tone.values())
                if new_state not in dp or dp[new_state][0] > diff:
                    dp[new_state] = (
                        diff,
                        [data[i]["name"], data[j]["name"]],
                        data[k]["name"],
                    )

# 最適解の探索
best_state = min(dp, key=lambda x: dp[x][0])

# 結果の表示
if dp[best_state][1] is not None:
    print("最適な組み合わせ:")
    print("下地:", dp[best_state][1][0])
    print("ファンデーション:", dp[best_state][1][1])
    print("コンシーラー:", dp[best_state][1][1])
    print("最終的な肌のトーン:", best_state)
    print("目標との差の最小距離:", np.sqrt(dp[best_state][0]))
else:
    print("最適な組み合わせが見つかりませんでした。")
