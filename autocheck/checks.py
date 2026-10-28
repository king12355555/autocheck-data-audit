# -*- coding: utf-8 -*-
"""确定性检查引擎。

核心原则: 所有数字都由代码计算, 大模型只负责“把算好的事实讲成人话”。
每个检查项返回一个 section:
    {key, title, status, summary, items: [{level, text}], advice: [str],
     data: {counts: {...}, frames: [{title, df, kind: rows|table}]}}

data.frames 用于“直接提取显示有问题数据”:
  - kind='rows'  : 具体有问题的原始记录(给用户看是哪几行出问题)
  - kind='table' : 汇总表(趋势/阈值/漏斗)
"""
import numpy as np
import pandas as pd

from .config import FUNNEL_STAGES


def _fmt_money(v):
    return f"{v:,.0f}"


def _fmt_signed(v):
    return f"{v:+,.0f}"


def _fmt_pct(v):
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------- 工具函数
def _pick_cols(df, cols):
    """取“人话”列顺序: 关键字段优先, 再补维度列。"""
    order = []
    for role in ("date", "order_id", "user_id", "amount", "status"):
        c = cols.get(role)
        if c and c in df.columns and c not in order:
            order.append(c)
    for c in cols.get("dimensions", []):
        if c in df.columns and c not in order:
            order.append(c)
    return order or list(df.columns[:6])


def _with_rownum(sub, keep, n=500):
    """给明细表加“文件行号”列: 表头为第 1 行, 第一条数据在第 2 行。"""
    out = sub[keep].head(n).copy()
    out.insert(0, "文件行号", (out.index + 2).astype(int))
    return out.reset_index(drop=True)


def _disp_rows(df, cols, mask, n=500, sort_by=None):
    """提取满足条件的原始行(带文件行号)。"""
    sub = df[mask.fillna(False) if hasattr(mask, "fillna") else mask]
    if sort_by and sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)
    return _with_rownum(sub, _pick_cols(df, cols), n=n)


def _new_section(key, title):
    return {"key": key, "title": title, "status": "ok",
            "summary": "", "items": [], "advice": [],
            "data": {"counts": {}, "frames": []}}


# ---------------------------------------------------------------- 数据质量
def check_data_quality(df, cols, cfg):
    chk = cfg["checks"]
    sec = _new_section("data_quality", "一、数据质量检查")
    items, advice = sec["items"], sec["advice"]
    data = sec["data"]
    counts = data["counts"]

    def add(lv, text):
        items.append({"level": lv, "text": text})
        if lv != "ok" and sec["status"] == "ok":
            sec["status"] = lv

    # 缺失率
    miss = df.isna().mean() * 100
    bad = miss[miss > chk["missing_alert_pct"]]
    for col, pct in bad.items():
        add("warn", f"列「{col}」缺失率 {pct:.1f}%, 超过阈值 {chk['missing_alert_pct']:.0f}%")

    # 完全重复行
    ndup = int(df.duplicated().sum())
    counts["dup_rows"] = ndup
    if ndup:
        add("error", f"存在 {ndup} 行完全重复记录")
        advice.append("完全重复行通常是取数时多表 join 导致, 建议先按业务主键去重再分析")

    # 金额检查 + 问题明细
    if cols.get("amount"):
        amt_col = cols["amount"]
        amt = pd.to_numeric(df[amt_col], errors="coerce")

        neg_mask = amt < 0
        n_neg = int(neg_mask.sum())
        counts["negative"] = n_neg
        if n_neg:
            add("error", f"赔付金额存在 {n_neg} 条负数记录, 需确认是否为冲正/退费误入")
            data["frames"].append({"title": f"负数金额记录(共 {n_neg} 条)",
                                   "df": _disp_rows(df, cols, neg_mask), "kind": "rows"})

        n_zero = int((amt == 0).sum())
        if n_zero:
            add("warn", f"赔付金额为 0 的记录 {n_zero} 条, 需确认是否有效赔付")

        n_na = int(amt.isna().sum())
        counts["nan_amount"] = n_na
        if n_na:
            add("warn", f"赔付金额缺失(空值) {n_na} 条, 无法参与金额汇总, 建议回源补数")
            data["frames"].append({"title": f"金额空值记录(共 {n_na} 条)",
                                   "df": _disp_rows(df, cols, amt.isna()), "kind": "rows"})

        valid = amt.dropna()
        if len(valid):
            mean = valid.mean()
            std = valid.std()
            outlier_thr = max(mean + chk["outlier_sigma"] * std, valid.quantile(0.999))
            out_mask = valid > outlier_thr
            n_out = int(out_mask.sum())
            counts["outliers"] = n_out
            if n_out:
                tops = valid[out_mask].nlargest(3)
                top_txt = ", ".join(f"{_fmt_money(v)}元" for v in tops)
                add("error", f"极端大额 {n_out} 条(超过均值+{chk['outlier_sigma']:.0f}σ 或 P99.9), 最大几条: {top_txt}")
                data["frames"].append({"title": f"极端大额记录(共 {n_out} 条)",
                                       "df": _disp_rows(df, cols, amt.isin(valid[out_mask].index), sort_by=amt_col),
                                       "kind": "rows"})
                advice.append("极端大额需逐条核对: 是正常高额赔付、数据错误, 还是套利/内部风险信号")

    # 关键ID为空
    for role, kw in (("order_id", "订单号"), ("user_id", "用户ID")):
        c = cols.get(role)
        if c:
            blank = df[c].isna() | (df[c].astype(str).str.strip() == "")
            n_blank = int(blank.sum())
            if role == "user_id":
                counts["blank_user"] = n_blank
            if n_blank:
                add("warn", f"{kw}列存在 {n_blank} 条空值, 将无法参与重复赔付/关联分析")
                if role == "user_id":
                    data["frames"].append({"title": f"{kw}空值记录(共 {n_blank} 条)",
                                           "df": _disp_rows(df, cols, blank), "kind": "rows"})

    if not items:
        add("ok", "未发现明显数据质量问题(缺失率、重复行、负金额、极端值均在阈值内)")
    sec["summary"] = f"发现问题 {len([i for i in items if i['level'] != 'ok'])} 项"
    return sec


# ---------------------------------------------------------------- 波动与突变
def _daily_series(df, date_col, amount_col):
    s = pd.to_numeric(df[amount_col], errors="coerce")
    d = pd.Series(s.values, index=pd.to_datetime(df[date_col], errors="coerce"))
    d = d.dropna()
    return d.groupby(level=0).sum().sort_index()


def _weekly_series(daily, drop_partial=True):
    ws = daily.index - pd.to_timedelta(daily.index.dayofweek, unit="D")
    wk = daily.groupby(ws).sum().sort_index()
    # 末尾若是不完整自然周(数据未覆盖到周日), 剔除, 避免“最后一周金额小”的假突变
    if drop_partial and len(wk):
        if wk.index[-1] + pd.Timedelta(days=6) > daily.index.max():
            wk = wk.iloc[:-1]
    return wk


def check_trend_mutation(daily, cfg):
    chk = cfg["checks"]
    thr = chk["wow_threshold"]
    sec = _new_section("trend", "二、趋势与突变检查")
    items, advice = sec["items"], sec["advice"]
    data = sec["data"]

    def add(lv, text):
        items.append({"level": lv, "text": text})
        if lv != "ok" and sec["status"] == "ok":
            sec["status"] = lv

    weekly = _weekly_series(daily)
    if len(weekly) < 3:
        sec["summary"] = "数据周期不足 3 周, 跳过周环比突变检测"
        return sec

    prev = weekly.shift(1)
    wow = weekly / prev - 1
    delta = weekly - prev
    flag = (prev > 0) & (wow.abs() >= thr)
    flagged = weekly[flag]

    add("ok", f"数据覆盖 {len(weekly)} 个自然周, 周赔付金额区间 [{_fmt_money(weekly.min())}, {_fmt_money(weekly.max())}] 元")
    if len(flagged):
        for wk_s, val in flagged.sort_values(ascending=False).head(5).items():
            w = wow[wk_s] * 100
            d = delta[wk_s]
            add("error", f"周({wk_s.date()})赔付金额 {_fmt_money(val)} 元, 环比上周 {'+' if d >= 0 else ''}{w:.1f}%"
                         f"({_fmt_signed(d)} 元), 超过 ±{thr * 100:.0f}% 突变阈值")
        advice.append("突变排查顺序: ①规则/产品上线(对齐发布日志) ②外部因素(节假日/活动) ③系统问题 ④套利/重复赔付激增")
    else:
        add("ok", f"最近 {len(weekly)} 周内未出现超过 ±{thr * 100:.0f}% 的周环比突变, 整体平稳")

    # 汇总表: 近16周 金额 + 环比
    wdf = pd.DataFrame({"周起始(周一)": weekly.index.strftime("%Y-%m-%d"),
                        "赔付金额": weekly.values.astype(int),
                        "环比%": (wow * 100).round(1)})
    wdf["标记"] = np.where((wow.abs() >= thr) & prev.values > 0, "⚠️ 突变", "")
    data["counts"]["big_weeks"] = int(len(flagged))
    data["frames"].append({"title": f"每周赔付金额趋势(近 {min(len(wdf), 16)} 周)",
                           "df": wdf.tail(16).iloc[::-1].reset_index(drop=True), "kind": "table"})
    sec["summary"] = "周环比突变" + ("有告警" if len(flagged) else "无告警")
    return sec


# ---------------------------------------------------------------- 归因分析
def _window_rows(df, date_col, start, end):
    t = pd.to_datetime(df[date_col], errors="coerce")
    return df[(t >= start) & (t < end)].copy()


def check_attribution(df, date_col, amount_col, dims, daily, cfg):
    """定位最大突变周(峰期), 与上一周(基期)对比做量价分解 + 维度下钻 + 峰期问题单提取。"""
    chk = cfg["checks"]
    thr = chk["wow_threshold"]
    sec = _new_section("attribution", "三、波动归因")
    items, advice = sec["items"], sec["advice"]
    data = sec["data"]
    counts = data["counts"]

    def add(lv, text):
        items.append({"level": lv, "text": text})
        if lv != "ok" and sec["status"] == "ok":
            sec["status"] = lv

    weekly = _weekly_series(daily)
    if len(weekly) < 3:
        sec["summary"] = "数据不足, 跳过归因"
        return sec

    prev = weekly.shift(1)
    delta = weekly - prev
    flag = (prev > 0) & ((weekly / prev - 1).abs() >= thr)
    flagged = weekly[flag]

    if len(flagged) == 0:
        if dims:
            g = df.groupby(dims[0])[amount_col].sum().sort_values(ascending=False)
            top = g.head(3)
            share = g.sum()
            txt = ", ".join(f"{k} {_fmt_money(v)}元({_fmt_pct(v / share)})" for k, v in top.items())
            items.append({"level": "ok", "text": f"本期无显著周环比突变, 按「{dims[0]}」看金额结构: {txt}"})
        else:
            items.append({"level": "ok", "text": "本期无显著周环比突变, 且无维度列可下钻"})
        sec["summary"] = "无显著突变, 仅输出结构"
        return sec

    peak_ws = flagged.sort_values(ascending=False).index[0]
    peak_start, peak_end = peak_ws, peak_ws + pd.Timedelta(days=7)
    base_start, base_end = peak_start - pd.Timedelta(days=7), peak_start
    peak = _window_rows(df, date_col, peak_start, peak_end)
    base = _window_rows(df, date_col, base_start, base_end)

    amt_p = pd.to_numeric(peak[amount_col], errors="coerce").sum()
    amt_b = pd.to_numeric(base[amount_col], errors="coerce").sum()
    cnt_p, cnt_b = len(peak), len(base)
    avg_p = amt_p / cnt_p if cnt_p else 0
    avg_b = amt_b / cnt_b if cnt_b else 0
    d_amt = amt_p - amt_b
    vol_effect = (cnt_p - cnt_b) * avg_b
    price_effect = cnt_p * (avg_p - avg_b)

    counts["has_peak"] = True
    counts["peak_week"] = peak_start.strftime("%Y-%m-%d")
    counts["delta_money"] = int(round(d_amt))
    counts["delta_pct"] = float(d_amt / amt_b) if amt_b else 0.0

    items.append({"level": "warn",
                  "text": f"峰期: 周({peak_start.date()})金额 {_fmt_money(amt_p)} 元 / {cnt_p} 单; "
                          f"基期(上周): {_fmt_money(amt_b)} 元 / {cnt_b} 单"})
    items.append({"level": "warn",
                  "text": f"量价拆解: 总增量 {_fmt_money(d_amt)} 元 = 单量效应 {_fmt_money(vol_effect)} 元"
                          f"({'占比 ' + _fmt_pct(vol_effect / d_amt) if d_amt else ''}) + 单均效应 {_fmt_money(price_effect)} 元"})

    main_dim_txt = []
    for dim in dims[:2]:
        g_p = pd.to_numeric(peak[amount_col], errors="coerce").groupby(peak[dim]).sum()
        g_b = pd.to_numeric(base[amount_col], errors="coerce").groupby(base[dim]).sum()
        keys = g_p.index.union(g_b.index)
        dd = pd.DataFrame({"peak": g_p.reindex(keys).fillna(0), "base": g_b.reindex(keys).fillna(0)})
        dd["delta"] = dd["peak"] - dd["base"]
        dd = dd.sort_values("delta", ascending=False)
        top = dd.head(5)
        txt = "; ".join(f"{k}: {_fmt_signed(r['delta'])}元(峰期 {_fmt_money(r['peak'])} 元)" for k, r in top.iterrows())
        main = top.index[0]
        items.append({"level": "warn", "text": f"按「{dim}」下钻, 增量主要来自: {txt}"})
        items.append({"level": "warn",
                      "text": f"主因判断: 「{dim} = {main}」贡献了该维度最大增量, 建议优先核查此分组"})
        main_dim_txt.append(f"{dim}={main}")
        advice.append(f"针对「{dim}={main}」: 核查该分组对应业务(供应商/渠道/规则)在峰期是否有变更")
        # 下钻小表
        tbl = pd.DataFrame({"取值": top.index.astype(str), "峰期金额": top["peak"].astype(int),
                            "较上周增量": top["delta"].astype(int)})
        data["frames"].append({"title": f"按「{dim}」下钻(增量 Top5)", "df": tbl.reset_index(drop=True),
                               "kind": "table"})

    if len(main_dim_txt) >= 1:
        counts["main_reason"] = main_dim_txt[0].split("=", 1)[1] if dims else ""
    if len(main_dim_txt) >= 2:
        counts["main_channel"] = main_dim_txt[1].split("=", 1)[1]

    # 峰期“问题单”: 金额最大的具体记录(便于直接查看是哪些订单, 带文件行号)
    peak = peak.copy()
    peak["_amt"] = pd.to_numeric(peak[amount_col], errors="coerce")
    peak = peak.sort_values("_amt", ascending=False).drop(columns=["_amt"])
    cols = {"date": date_col, "amount": amount_col, "dimensions": dims,
            "order_id": None, "user_id": None, "status": None}
    peak_rows = _with_rownum(peak, _pick_cols(peak, cols), n=10)
    counts["peak_rows_n"] = len(peak_rows)
    data["frames"].append({"title": f"峰期问题单({peak_start.date()} 那周金额最大的 10 条)",
                           "df": peak_rows, "kind": "rows"})

    advice.append("归因结论需抽样拉明细人工复核, 确认因果而非仅相关性(口径一致前提下)")
    sec["summary"] = f"峰期 {peak_start.date()} 增量主因已定位"
    sec["status"] = "warn"
    return sec


# ---------------------------------------------------------------- 动态阈值
def check_threshold_alerts(daily, cfg):
    chk = cfg["checks"]
    win = int(chk["rolling_days"])
    k = float(chk["zscore_k"])
    sec = _new_section("threshold", "四、动态阈值预警")
    items, advice = sec["items"], sec["advice"]
    data = sec["data"]

    def add(lv, text):
        items.append({"level": lv, "text": text})
        if lv != "ok" and sec["status"] == "ok":
            sec["status"] = lv

    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    if len(full.dropna()) < win * 2:
        sec["summary"] = "有效天数不足, 跳过"
        return sec

    roll_mean = full.rolling(win, min_periods=win).mean()
    roll_std = full.rolling(win, min_periods=win).std()
    upper = roll_mean + k * roll_std
    alerts = full[(full > upper) & (roll_std > 0)].dropna()

    alert_rows = []
    for ts, v in alerts.items():
        alert_rows.append({"日期": ts.strftime("%Y-%m-%d"), "当日赔付金额": int(round(v)),
                           "阈值上限": int(round(upper[ts])),
                           "高出阈值%": round((v / upper[ts] - 1) * 100, 1)})
    if alert_rows:
        data["frames"].append({"title": f"触发动态阈值的日期(共 {len(alert_rows)} 天)",
                               "df": pd.DataFrame(alert_rows), "kind": "table"})

    if len(alerts):
        rows = alerts.tail(8)
        for ts, v in rows.items():
            ex = v / upper[ts] - 1
            add("error", f"{ts.date()} 赔付金额 {_fmt_money(v)} 元, 超过滚动均值+{k:.0f}σ 上限"
                         f"({_fmt_money(upper[ts])} 元, 高出上限 {_fmt_pct(ex)})")
        add("ok", f"共 {len(alerts)} 个自然日触发动态阈值(近 {win} 日滚动窗口), 上表为最近 {len(rows)} 个")
        advice.append("动态阈值按场景分桶更准(不同渠道/原因量级不同), 节假日需单独放宽; 上线后回看误报率与漏报率")
    else:
        add("ok", f"近 {win} 日滚动均值±{k:.0f}σ 阈值下, 无自然日超限, 波动处于正常区间")

    data["counts"]["alert_days"] = int(len(alerts))
    sec["summary"] = f"触发 {len(alerts)} 天" if len(alerts) else "无超限"
    return sec


# ---------------------------------------------------------------- 重复/套利
def check_duplicates(df, cols, cfg):
    chk = cfg["checks"]
    sec = _new_section("duplicates", "五、重复赔付与高频申请")
    items, advice = sec["items"], sec["advice"]
    data = sec["data"]
    counts = data["counts"]
    order_c, user_c, amt_c, date_c = cols.get("order_id"), cols.get("user_id"), cols.get("amount"), cols.get("date")

    def add(lv, text):
        items.append({"level": lv, "text": text})
        if lv != "ok" and sec["status"] == "ok":
            sec["status"] = lv

    if order_c:
        dup = df.groupby(order_c).size()
        dup_orders = dup[dup > 1]
        if len(dup_orders):
            sub = df[df[order_c].isin(dup_orders.index)]
            extra = int(dup_orders.sum() - len(dup_orders))
            tot = pd.to_numeric(sub[amt_c], errors="coerce").sum() if amt_c else 0
            counts["dup_orders"] = int(len(dup_orders))
            counts["dup_extra"] = extra
            examples = dup_orders.sort_values(ascending=False).head(5)
            ex_txt = ", ".join(f"{oid}({c}次)" for oid, c in examples.items())
            add("error", f"发现 {len(dup_orders)} 个订单赔付超过 1 次, 涉及 {extra} 条疑似重复赔付记录"
                         f"{f', 涉及金额约 {_fmt_money(tot)} 元' if amt_c else ''}")
            add("warn", f"重复最严重的订单示例: {ex_txt}")
            # 明细: 这些重复订单的完整记录(带文件行号)
            sub = sub.sort_values([order_c, date_c] if date_c else [order_c])
            data["frames"].append({"title": f"重复赔付订单的完整记录(共 {len(sub)} 条)",
                                   "df": _with_rownum(sub, _pick_cols(sub, cols)), "kind": "rows"})
            advice.append("同订单重复赔付需核查: 是系统重试/重复打款, 还是套利行为; 建议按业务主键加唯一约束")
        else:
            add("ok", f"未发现同订单重复赔付(共检查 {len(dup)} 个订单)")
    else:
        sec["summary"] = "缺少订单号列, 跳过重复赔付检测"
        return sec

    if user_c and date_c:
        try:
            d = df[[user_c, date_c]].copy()
            d[date_c] = pd.to_datetime(d[date_c], errors="coerce")
            d = d.dropna(subset=[user_c, date_c])
            d = d[d[user_c].astype(str).str.strip() != ""]
            d = d.sort_values([user_c, date_c])
            win_days = int(chk["dup_user_window_days"])
            min_cnt = int(chk["dup_user_min_count"])
            d["_sec"] = d[date_c].values.astype("datetime64[ns]").astype("int64") // 10**9

            def _wincount(s):
                t = s.values
                lo = t - win_days * 86400
                left = np.searchsorted(t, lo, side="left")
                right = np.searchsorted(t, t, side="right")
                return pd.Series(right - left, index=s.index)

            d["_cnt"] = d.groupby(user_c, sort=False)["_sec"].transform(_wincount)
            flagged = d[d["_cnt"] >= min_cnt]
            n_users = flagged[user_c].nunique()
            counts["high_freq_users"] = int(n_users)
            if n_users:
                by_user = flagged.groupby(user_c)["_cnt"].max().sort_values(ascending=False)
                top = by_user.head(5)
                ex_txt = ", ".join(f"{u}(窗口内{int(c)}次)" for u, c in top.items())
                add("warn", f"同一用户 {win_days} 天内申请 ≥{min_cnt} 次的疑似高频/套利用户 {n_users} 个")
                add("warn", f"最高频用户示例: {ex_txt}")
                # 明细: 高频用户的全部申请记录
                hot_users = top.index.tolist()
                sub2 = df[df[user_c].isin(hot_users)].copy()
                sub2 = sub2.sort_values([user_c, date_c])
                data["frames"].append({"title": f"高频用户的申请记录(共 {len(sub2)} 条)",
                                       "df": _with_rownum(sub2, _pick_cols(sub2, cols)), "kind": "rows"})
                advice.append("高频申请需结合设备指纹/银行卡/地址相似度综合判断, 先抽样人工复核再处理, 避免误伤正常用户")
            else:
                add("ok", f"未发现同一用户 {win_days} 天内申请 ≥{min_cnt} 次的高频行为")
        except Exception as e:  # noqa
            add("ok", f"高频申请检测跳过(数据格式问题: {e})")
    else:
        add("ok", "缺少用户ID或时间列, 高频申请检测跳过")

    sec["summary"] = "有疑似重复/高频" if sec["status"] != "ok" else "未发现异常"
    return sec


# ---------------------------------------------------------------- 漏斗
def _match_stage(uniq, keywords):
    for kw in keywords:
        for u in uniq:
            if kw.lower() in str(u).lower():
                return u
    return None


def check_funnel(df, status_col, amount_col):
    sec = _new_section("funnel", "六、赔付流程漏斗")
    items, advice = sec["items"], sec["advice"]
    data = sec["data"]

    def add(lv, text):
        items.append({"level": lv, "text": text})
        if lv != "ok" and sec["status"] == "ok":
            sec["status"] = lv

    if not status_col:
        sec["summary"] = "缺少状态列, 跳过漏斗"
        return sec
    uniq = list(df[status_col].dropna().unique())
    stages = []
    for kw_group in FUNNEL_STAGES:
        s = _match_stage(uniq, kw_group)
        if s and s not in stages:
            stages.append(s)
    if len(stages) < 2:
        sec["summary"] = "状态取值无法匹配到漏斗阶段, 跳过"
        return sec

    # 状态列为“当前/最终所处环节”: 达到环节i = 状态在环节i及更深环节的行数之和(从后往前累计)
    counts = [int((df[status_col] == s).sum()) for s in stages]
    total = len(df)
    unmatched = total - sum(counts)
    suffix = []
    acc = 0
    for c in reversed(counts):
        acc += c
        suffix.append(acc)
    suffix.reverse()

    add("ok", f"环节1「赔付申请(全部)」: {total} 单")
    prev = total
    for i in range(1, len(stages)):
        n = suffix[i]
        conv = n / prev if prev else 0
        drop = 1 - conv
        lv = "ok" if drop < 0.15 else ("warn" if drop < 0.3 else "error")
        if lv != "ok" and sec["status"] == "ok":
            sec["status"] = lv
        add(lv, f"环节{i + 1}「{stages[i]}」: 累计 {n} 单, 较上一环节转化率 {_fmt_pct(conv)}"
                f"(流失 {_fmt_pct(drop)})")
        prev = n
    overall = suffix[-1] / total if total else 0
    add("ok", f"整体通过率: {_fmt_pct(overall)} ({len(stages)} 个环节)")
    if unmatched:
        add("warn", f"另有 {unmatched} 条记录的状态不在漏斗环节取值内, 未纳入漏斗")
    advice.append("对流失最大的环节下钻拒因分布(资料/规则/误拒), 规则漏洞常藏在流失异常的环节里")

    # 漏斗表
    tbl_rows = [{"环节": "赔付申请(全部)", "累计单数": total, "转化率": ""}]
    for i in range(1, len(stages)):
        conv = suffix[i] / suffix[i - 1] if suffix[i - 1] else None
        tbl_rows.append({"环节": f"{stages[i]}", "累计单数": suffix[i],
                         "转化率": f"{conv * 100:.1f}%" if conv is not None else ""})
    data["frames"].append({"title": "赔付漏斗各环节(累计到达数)", "df": pd.DataFrame(tbl_rows), "kind": "table"})
    data["counts"]["overall_rate"] = float(overall)
    sec["summary"] = f"整体通过率 {_fmt_pct(overall)}"
    return sec


# ---------------------------------------------------------------- 汇总
def run_all_checks(df, cols, cfg):
    date_c, amt_c = cols.get("date"), cols.get("amount")
    sections = []
    daily = None

    # 数据概览
    sec = _new_section("overview", "〇、数据概览")
    sec["items"].append({"level": "ok", "text": f"总记录数: {len(df)} 行, 字段数: {len(df.columns)} 个"})
    meta = {"rows": len(df), "cols_count": len(df.columns)}
    if date_c:
        t = pd.to_datetime(df[date_c], errors="coerce")
        valid = t.dropna()
        meta["date_min"], meta["date_max"] = valid.min(), valid.max()
        sec["items"].append({"level": "ok", "text": f"时间范围: {meta['date_min'].date()} ~ {meta['date_max'].date()}"})
    if amt_c:
        amt = pd.to_numeric(df[amt_c], errors="coerce")
        meta["amount_total"] = float(amt.sum())
        meta["amount_avg"] = float(amt.mean())
        sec["items"].append({"level": "ok",
                             "text": f"赔付总额: {_fmt_money(meta['amount_total'])} 元; 单均: {_fmt_money(meta['amount_avg'])} 元"})
    if date_c and amt_c:
        daily = _daily_series(df, date_c, amt_c)
    sec["summary"] = f"{len(df)} 行"
    sections.append(sec)

    sections.append(check_data_quality(df, cols, cfg))
    if daily is not None:
        sections.append(check_trend_mutation(daily, cfg))
        sections.append(check_attribution(df, date_c, amt_c, cols.get("dimensions", []), daily, cfg))
        sections.append(check_threshold_alerts(daily, cfg))
    else:
        sections.append({"key": "trend", "title": "二、趋势与突变检查", "status": "ok",
                         "summary": "缺少时间+金额列, 跳过", "items": [],
                         "advice": [], "data": {"counts": {}, "frames": []}})
        sections.append({"key": "threshold", "title": "四、动态阈值预警", "status": "ok",
                         "summary": "缺少时间+金额列, 跳过", "items": [],
                         "advice": [], "data": {"counts": {}, "frames": []}})
    sections.append(check_duplicates(df, cols, cfg))
    sections.append(check_funnel(df, cols.get("status"), amt_c))
    return sections, meta
