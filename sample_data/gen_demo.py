# -*- coding: utf-8 -*-
"""生成模拟赔付明细数据(内置 6 类问题, 用于验证 AutoCheck 的检出能力)。

注入的问题(ground truth):
  1. 突变周: 2026-04-06 起一周内「到店无房」赔付单量×2 且单均×1.5 → 周金额应显著超过 ±25% 阈值
  2. 重复赔付: 10 个订单在 7 天内被再次赔付
  3. 高频用户: 5 个用户 30 天内申请 ≥3 次
  4. 负数金额: 8 条
  5. 金额缺失: 6 条 / 用户ID缺失: 5 条
  6. 极端大额: 3 条(5万~8万元)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

START, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-08-31")
REASONS = [("到店无房", 0.06, 620), ("预订信息不符", 0.12, 330), ("卫生与设施问题", 0.20, 160),
           ("服务态度问题", 0.26, 90), ("价格问题", 0.14, 130), ("行程变更", 0.22, 110)]
CHANNELS = [("APP", 0.52), ("小程序", 0.28), ("电话客服", 0.12), ("PC", 0.08)]
STARS = [("经济型", 0.40), ("舒适型", 0.34), ("高档型", 0.18), ("豪华型", 0.08)]


def draw_amount(avg):
    return max(20, round(rng.gamma(shape=1.7, scale=avg / 1.7)))


def draw_status():
    """链式漏斗: 已申请(100%) → 资料齐全(96.5%) → 审核通过(90%) → 已打款(97%)"""
    if rng.random() < 0.965:
        if rng.random() < 0.90:
            if rng.random() < 0.97:
                return "已打款"
            return "审核通过"
        return "资料齐全"
    return "已申请"


def main():
    days = pd.date_range(START, END, freq="D")
    rows = []
    seq = 0
    for day in days:
        wf = 1.35 if day.dayofweek >= 5 else 1.0
        n = int(rng.poisson(46 * wf))
        for _ in range(n):
            seq += 1
            reason, rp, avg = REASONS[rng.choice(len(REASONS), p=[p for _, p, _ in REASONS])]
            channel, cp = CHANNELS[rng.choice(len(CHANNELS), p=[p for _, p in CHANNELS])]
            star, sp = STARS[rng.choice(len(STARS), p=[p for _, p in STARS])]
            rows.append({
                "订单号": f"O{day.strftime('%Y%m%d')}{seq:05d}",
                "用户ID": f"U{rng.integers(100000, 999999)}",
                "赔付申请时间": day.strftime("%Y-%m-%d"),
                "赔付金额": draw_amount(avg),
                "赔付原因": reason,
                "渠道": channel,
                "酒店星级": star,
                "状态": draw_status(),
            })
    df = pd.DataFrame(rows)
    print("基线生成:", len(df), "行")

    # ---------- 注入 1: 突变周(2026-04-06 ~ 04-12), 到店无房 单量×2、单均×1.5 ----------
    # 用“新增真实订单”实现(不复制原订单号, 避免制造完全重复行假象)
    wk = pd.Timestamp("2026-04-06")
    mask = (df["赔付申请时间"] >= str(wk.date())) & (df["赔付申请时间"] <= str(wk.date() + pd.Timedelta(days=6))) \
           & (df["赔付原因"] == "到店无房")
    n_base = int(mask.sum())
    add_rows = []
    for _, r in df.loc[mask].iterrows():
        seq += 1
        d = pd.Timestamp(r["赔付申请时间"])
        add_rows.append({
            "订单号": f"O{d.strftime('%Y%m%d')}{seq:05d}",
            "用户ID": f"U{rng.integers(100000, 999999)}",
            "赔付申请时间": d.strftime("%Y-%m-%d"),
            "赔付金额": draw_amount(930),  # 620 × 1.5
            "赔付原因": "到店无房",
            "渠道": rng.choice([c for c, _ in CHANNELS]),
            "酒店星级": rng.choice([s for s, _ in STARS]),
            "状态": draw_status(),
        })
    df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)
    print(f"注入突变周: 到店无房 {n_base} -> {n_base * 2} 单, 新增单均约930元")

    # ---------- 注入 2: 同订单 7 天内重复赔付(10 个) ----------
    base_orders = df.sample(1200, random_state=7)
    dup_rows = []
    for _, r in base_orders.head(10).iterrows():
        d = pd.Timestamp(r["赔付申请时间"]) + pd.Timedelta(days=int(rng.integers(1, 6)))
        if d > END:
            d = END
        dup_rows.append({
            "订单号": r["订单号"], "用户ID": r["用户ID"],
            "赔付申请时间": d.strftime("%Y-%m-%d"),
            "赔付金额": int(rng.integers(80, 220)), "赔付原因": rng.choice(["价格问题", "行程变更"]),
            "渠道": r["渠道"], "酒店星级": r["酒店星级"], "状态": "已打款",
        })
    df = pd.concat([df, pd.DataFrame(dup_rows)], ignore_index=True)
    print("注入重复赔付订单:", len(dup_rows))

    # ---------- 注入 3: 5 个用户 30 天内高频申请(各 +4 单) ----------
    users = df["用户ID"].value_counts()
    first_date = df.groupby("用户ID")["赔付申请时间"].min()
    cand = first_date[first_date < "2026-07-01"].index.values
    picked = list(rng.choice(cand, size=5, replace=False))
    hi_rows = []
    for u in picked:
        first = df.loc[df["用户ID"] == u, "赔付申请时间"].min()
        d0 = pd.Timestamp(first)
        for off in [3, 9, 16, 23]:
            d = d0 + pd.Timedelta(days=off)
            if d > END:
                d = END
            hi_rows.append({
                "订单号": f"HF{d.strftime('%Y%m%d')}{rng.integers(0, 9999):04d}", "用户ID": u,
                "赔付申请时间": d.strftime("%Y-%m-%d"),
                "赔付金额": int(rng.integers(50, 160)), "赔付原因": "价格问题",
                "渠道": rng.choice([c for c, _ in CHANNELS]), "酒店星级": "经济型", "状态": "已打款",
            })
    df = pd.concat([df, pd.DataFrame(hi_rows)], ignore_index=True)
    print("注入高频用户:", len(picked), "个, 各+4单")

    # ---------- 注入 4/5/6: 脏数据 ----------
    junk = []
    for _ in range(8):
        junk.append({"订单号": f"NG{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-06-15", "赔付金额": -int(rng.integers(20, 150)),
                     "赔付原因": "行程变更", "渠道": "APP", "酒店星级": "经济型", "状态": "已申请"})
    for _ in range(6):
        junk.append({"订单号": f"NA{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-06-16", "赔付金额": np.nan,
                     "赔付原因": "服务态度问题", "渠道": "小程序", "酒店星级": "舒适型", "状态": "已申请"})
    for _ in range(5):
        junk.append({"订单号": f"NU{rng.integers(10**6, 10**7)}", "用户ID": "",
                     "赔付申请时间": "2026-06-17", "赔付金额": 120,
                     "赔付原因": "价格问题", "渠道": "APP", "酒店星级": "经济型", "状态": "资料齐全"})
    # 极端大额并入突变周(2026-04-06 ~ 04-12): 真实场景中大额异常单常常就是波动主因
    for v, dd in ((30000, "2026-04-06"), (45000, "2026-04-08"), (60000, "2026-04-10")):
        junk.append({"订单号": f"OL{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": dd, "赔付金额": v,
                     "赔付原因": "到店无房", "渠道": "APP", "酒店星级": "豪华型", "状态": "已打款"})
    df = pd.concat([df, pd.DataFrame(junk)], ignore_index=True)
    print("注入脏数据:", 8 + 6 + 5 + 3, "条")

    df = df.sort_values("赔付申请时间").reset_index(drop=True)
    out = "sample_data/赔付明细_示例.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存: {out}  ({len(df)} 行)")


if __name__ == "__main__":
    main()
