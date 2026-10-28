# -*- coding: utf-8 -*-
"""生成 2 个带不同注入问题的测试样本(用于验证 AutoCheck 检出能力, 与 gen_demo.py 场景区分)。

样本B(seed=7): 渠道异常 + 套利专题
  注入: ① 电话客服×到店无房 2026-05-11 周爆发(单量×5、金额×1.5, 混入 26k/18k 大额)
        ② 8 个订单重复赔付(其中 1 个订单赔 3 次)  ③ 6 个用户 25 天内高频申请
        ④ 负数5 零金额15 金额空4 用户空6 完全重复行3
样本C(seed=11): 波动形态 + 漏斗专题
  注入: ① 2026-07-13 周全部赔付金额×1.3(模拟赔付标准普调, 单均效应≈100%)
        ② 审核环节通过率压低至 62%(漏斗审核流失≈38%)  ③ 3 条状态脏值“已赔付”
        ④ 负数3 零金额20 完全重复行5 用户空4 金额空3
"""
import numpy as np
import pandas as pd

REASONS = [("到店无房", 0.06, 620), ("预订信息不符", 0.12, 330), ("卫生与设施问题", 0.20, 160),
           ("服务态度问题", 0.26, 90), ("价格问题", 0.14, 130), ("行程变更", 0.22, 110)]


def draw_amount(rng, avg):
    # 正常赔付有业务额度上限(约 4 倍单均), 避免重尾分布把自然大额误判为“极端大额”
    return min(max(20, round(rng.gamma(shape=1.7, scale=avg / 1.7))), int(avg * 4))


def draw_status(rng, p_info=0.965, p_audit=0.90, p_pay=0.97):
    if rng.random() < p_info:
        if rng.random() < p_audit:
            if rng.random() < p_pay:
                return "已打款"
            return "审核通过"
        return "资料齐全"
    return "已申请"


def base_rows(rng, start, end, channel_probs, seed_extra=None, status_p=None):
    days = pd.date_range(start, end, freq="D")
    rows, seq = [], 0
    for day in days:
        wf = 1.35 if day.dayofweek >= 5 else 1.0
        n = int(rng.poisson(46 * wf))
        for _ in range(n):
            seq += 1
            reason, rp, avg = REASONS[rng.choice(len(REASONS), p=[p for _, p, _ in REASONS])]
            channel, cp = channel_probs[rng.choice(len(channel_probs), p=[p for _, p in channel_probs])]
            star, sp = ([("经济型", .4), ("舒适型", .34), ("高档型", .18), ("豪华型", .08)]
                        [rng.choice(4, p=[.4, .34, .18, .08])])
            rows.append({
                "订单号": f"O{day.strftime('%Y%m%d')}{seq:05d}",
                "用户ID": f"U{rng.integers(100000, 999999)}",
                "赔付申请时间": day.strftime("%Y-%m-%d"),
                "赔付金额": draw_amount(rng, avg),
                "赔付原因": reason, "渠道": channel, "酒店星级": star,
                "状态": (draw_status(rng, *status_p) if status_p else draw_status(rng)),
            })
    return pd.DataFrame(rows), seq


def save(df, path):
    df = df.sort_values("赔付申请时间").reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存:", path, f"({len(df)} 行)")


# ================================================================ 样本 B
def sample_b():
    rng = np.random.default_rng(7)
    CH = [("APP", .48), ("小程序", .20), ("电话客服", .22), ("PC", .10)]
    df, seq = base_rows(rng, pd.Timestamp("2025-10-01"), pd.Timestamp("2026-08-31"), CH)
    print("B 基线:", len(df), "行")

    # ① 电话客服×到店无房 2026-05-11 周爆发
    wk = pd.Timestamp("2026-05-11")
    m = (df["赔付申请时间"] >= str(wk.date())) & (df["赔付申请时间"] <= str(wk.date() + pd.Timedelta(days=6))) \
        & (df["赔付原因"] == "到店无房") & (df["渠道"] == "电话客服")
    n_base = int(m.sum())
    add = []
    for _, r in df.loc[m].iterrows():
        for _ in range(4):
            seq += 1
            d = pd.Timestamp(r["赔付申请时间"])
            add.append({"订单号": f"O{d.strftime('%Y%m%d')}{seq:05d}", "用户ID": f"U{rng.integers(100000, 999999)}",
                        "赔付申请时间": d.strftime("%Y-%m-%d"), "赔付金额": draw_amount(rng, 930),
                        "赔付原因": "到店无房", "渠道": "电话客服",
                        "酒店星级": rng.choice(["经济型", "舒适型", "高档型", "豪华型"]), "状态": draw_status(rng)})
    for v, dd in ((26000, "2026-05-12"), (18000, "2026-05-14")):
        seq += 1
        add.append({"订单号": f"O202605{seq:04d}", "用户ID": f"U{rng.integers(100000, 999999)}",
                    "赔付申请时间": dd, "赔付金额": v, "赔付原因": "到店无房", "渠道": "电话客服",
                    "酒店星级": "豪华型", "状态": "已打款"})
    df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
    print(f"  B-① 电话客服到店无房 {n_base} -> {n_base * 5 + 2} 单(含大额2条)")

    # ② 8 个订单二赔 + 1 个订单三赔
    dup_rows = []
    base_orders = df.sample(1500, random_state=7)
    for i, (_, r) in enumerate(base_orders.head(9).iterrows()):
        times = [1, 3] if i == 0 else [int(rng.integers(1, 6))]
        for off in times:
            d = pd.Timestamp(r["赔付申请时间"]) + pd.Timedelta(days=off)
            if d > pd.Timestamp("2026-08-31"):
                d = pd.Timestamp("2026-08-31")
            dup_rows.append({"订单号": r["订单号"], "用户ID": r["用户ID"], "赔付申请时间": d.strftime("%Y-%m-%d"),
                             "赔付金额": int(rng.integers(80, 220)), "赔付原因": rng.choice(["价格问题", "行程变更"]),
                             "渠道": r["渠道"], "酒店星级": r["酒店星级"], "状态": "已打款"})
    df = pd.concat([df, pd.DataFrame(dup_rows)], ignore_index=True)
    print("  B-② 重复赔付: 8个订单二赔 + 1个订单三赔 =", len(dup_rows), "条额外赔付")

    # ③ 6 个用户高频
    first_date = df.groupby("用户ID")["赔付申请时间"].min()
    cand = first_date[first_date < "2026-06-01"].index.values
    picked = list(rng.choice(cand, size=6, replace=False))
    hi = []
    for u in picked:
        d0 = pd.Timestamp(df.loc[df["用户ID"] == u, "赔付申请时间"].min())
        for off in [4, 11, 18]:
            d = d0 + pd.Timedelta(days=off)
            if d > pd.Timestamp("2026-08-31"):
                d = pd.Timestamp("2026-08-31")
            seq += 1
            hi.append({"订单号": f"HF{d.strftime('%Y%m%d')}{seq % 10000:04d}", "用户ID": u,
                       "赔付申请时间": d.strftime("%Y-%m-%d"), "赔付金额": int(rng.integers(50, 160)),
                       "赔付原因": "价格问题", "渠道": rng.choice([c for c, _ in CH]),
                       "酒店星级": "经济型", "状态": "已打款"})
    df = pd.concat([df, pd.DataFrame(hi)], ignore_index=True)
    print("  B-③ 高频用户 6 个, 各 +3 单")

    # ④ 脏数据
    junk = []
    for _ in range(5):
        junk.append({"订单号": f"NG{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-03-10", "赔付金额": -int(rng.integers(20, 150)),
                     "赔付原因": "行程变更", "渠道": "APP", "酒店星级": "经济型", "状态": "已申请"})
    for _ in range(15):
        junk.append({"订单号": f"ZR{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-04-20", "赔付金额": 0,
                     "赔付原因": "价格问题", "渠道": "APP", "酒店星级": "经济型", "状态": "资料齐全"})
    for _ in range(4):
        junk.append({"订单号": f"NA{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-05-01", "赔付金额": np.nan,
                     "赔付原因": "服务态度问题", "渠道": "小程序", "酒店星级": "舒适型", "状态": "已申请"})
    for _ in range(6):
        junk.append({"订单号": f"NU{rng.integers(10**6, 10**7)}", "用户ID": "",
                     "赔付申请时间": "2026-05-02", "赔付金额": 120,
                     "赔付原因": "价格问题", "渠道": "APP", "酒店星级": "经济型", "状态": "资料齐全"})
    for i in range(3):  # 完全重复行(复制前3行)
        junk.append(df.iloc[i].to_dict())
    df = pd.concat([df, pd.DataFrame(junk)], ignore_index=True)
    print("  B-④ 脏数据:", 5 + 15 + 4 + 6 + 3, "条")
    save(df, "sample_data/赔付样本B_渠道异常与套利.csv")


# ================================================================ 样本 C
def sample_c():
    rng = np.random.default_rng(11)
    CH = [("APP", .52), ("小程序", .28), ("电话客服", .12), ("PC", .08)]
    df, seq = base_rows(rng, pd.Timestamp("2025-09-01"), pd.Timestamp("2026-08-31"), CH,
                        status_p=(0.97, 0.62, 0.90))
    print("C 基线:", len(df), "行")

    # ① 2026-07-13 周 全量赔付金额 ×1.45(标准普调, 单均效应主导, 强度足以越过自然波动)
    wk = pd.Timestamp("2026-07-13")
    m = (df["赔付申请时间"] >= str(wk.date())) & (df["赔付申请时间"] <= str(wk.date() + pd.Timedelta(days=6)))
    df.loc[m, "赔付金额"] = (df.loc[m, "赔付金额"] * 1.45).round().astype(int)
    print(f"  C-① 07-13 周全量金额×1.45 (影响 {int(m.sum())} 单)")

    # ② 3 条状态脏值(全新订单号, 不在漏斗取值内; 避免复制造成假重复)
    for _ in range(3):
        seq += 1
        junk_status = {"订单号": f"BAD{seq:06d}", "用户ID": f"U{rng.integers(100000, 999999)}",
                       "赔付申请时间": "2026-07-20", "赔付金额": 200,
                       "赔付原因": "价格问题", "渠道": "APP", "酒店星级": "经济型", "状态": "已赔付"}
        df = pd.concat([df, pd.DataFrame([junk_status])], ignore_index=True)
    print("  C-② 状态脏值 3 条(新订单号)")

    # ③ 脏数据
    junk = []
    for _ in range(3):
        junk.append({"订单号": f"NG{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-02-10", "赔付金额": -int(rng.integers(20, 150)),
                     "赔付原因": "行程变更", "渠道": "APP", "酒店星级": "经济型", "状态": "已申请"})
    for _ in range(20):
        junk.append({"订单号": f"ZR{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-03-15", "赔付金额": 0,
                     "赔付原因": "价格问题", "渠道": "小程序", "酒店星级": "经济型", "状态": "资料齐全"})
    for _ in range(3):
        junk.append({"订单号": f"NA{rng.integers(10**6, 10**7)}", "用户ID": f"U{rng.integers(100000, 999999)}",
                     "赔付申请时间": "2026-06-20", "赔付金额": np.nan,
                     "赔付原因": "服务态度问题", "渠道": "APP", "酒店星级": "舒适型", "状态": "已申请"})
    for _ in range(4):
        junk.append({"订单号": f"NU{rng.integers(10**6, 10**7)}", "用户ID": "",
                     "赔付申请时间": "2026-06-21", "赔付金额": 150,
                     "赔付原因": "价格问题", "渠道": "APP", "酒店星级": "经济型", "状态": "资料齐全"})
    # 本样本刻意不注入: 完全重复行 / 重复赔付订单 / 高频用户(验证“没问题就不误报”)
    df = pd.concat([df, pd.DataFrame(junk)], ignore_index=True)
    print("  C-③ 脏数据:", 3 + 20 + 3 + 4, "条(无重复行/无重复订单/无高频)")
    save(df, "sample_data/赔付样本C_标准普调与审核流失.csv")


if __name__ == "__main__":
    sample_b()
    sample_c()
