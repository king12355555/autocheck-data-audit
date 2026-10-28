# -*- coding: utf-8 -*-
"""配置与列名自动识别。

设计原则:
- 列名通过关键词自动识别(支持中英文), 识别不到的检查项自动跳过并在报告中说明;
- 用户可通过 config.json 显式指定列名与阈值。
"""
import json

DEFAULT_CONFIG = {
    "table_name": "赔付明细",
    "columns": {
        "date": None,      # 时间列(赔付申请时间)
        "amount": None,    # 金额列(赔付金额)
        "order_id": None,  # 订单号列
        "user_id": None,   # 用户ID列
        "status": None,    # 状态列(漏斗用)
        "dimensions": []   # 维度列(归因下钻用), 如 ["赔付原因", "渠道", "酒店星级"]
    },
    "checks": {
        "wow_threshold": 0.25,          # 周环比突变阈值(±25%)
        "rolling_days": 7,              # 动态阈值滚动窗口
        "zscore_k": 2.0,                # 动态阈值: 均值 + k*标准差
        "dup_user_window_days": 30,     # 同用户高频申请窗口(天)
        "dup_user_min_count": 3,        # 窗口内最少申请次数(含本次)
        "missing_alert_pct": 5.0,       # 缺失率告警阈值(%)
        "outlier_sigma": 15.0,          # 极端值: 均值 + sigma倍标准差
    },
}

# 关键词组(按优先级), 用于列名自动识别
KEYWORDS = {
    "date": ["赔付申请时间", "申请时间", "赔付时间", "申请日期", "时间", "日期",
             "date", "pay_time", "apply_time", "created_at", "create_time", "time"],
    "amount": ["赔付金额", "补偿金额", "理赔金额", "金额", "amount", "pay_amount",
               "refund_amount", "compensation", "cost", "费用"],
    "order_id": ["订单号", "订单id", "订单编号", "订单", "order_id", "orderid",
                 "order_no", "order_no_", "order"],
    "user_id": ["用户id", "用户ID", "用户", "user_id", "userid", "uid",
                "customer_id", "member_id", "手机号", "身份证号"],
    "status": ["状态", "赔付状态", "审核状态", "stage", "status", "state"],
    "dimensions": [
        ["赔付原因", "原因", "reason", "refund_reason"],
        ["渠道", "channel", "来源", "source"],
        ["酒店星级", "星级", "star", "star_level", "酒店等级"],
        ["城市", "city"],
        ["酒店", "hotel", "hotel_name"],
    ],
}

# 漏斗阶段关键词(顺序即漏斗顺序)
FUNNEL_STAGES = [
    ["申请", "发起", "apply", "submit"],
    ["资料", "材料", "info", "document"],
    ["审核", "复核", "audit", "review", "check"],
    ["打款", "赔付完成", "放款", "pay", "paid", "finish"],
]


def load_config(path=None):
    """加载配置(深合并用户配置到默认配置)。"""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        cfg = _deep_merge(cfg, user)
    return cfg


def _deep_merge(base, override):
    out = json.loads(json.dumps(base))
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _find_col(df, keywords, explicit):
    if explicit:
        for c in df.columns:
            if str(c).strip() == str(explicit).strip():
                return c
        return None
    # 1) 完全匹配
    for kw in keywords:
        for c in df.columns:
            if str(kw).lower() == str(c).lower():
                return c
    # 2) 包含匹配(取最短命中, 减少误伤)
    hits = []
    for kw in keywords:
        for c in df.columns:
            if str(kw).lower() in str(c).lower():
                hits.append(c)
    if hits:
        return min(hits, key=len)
    return None


def resolve_columns(df, cfg):
    """自动识别列映射。返回 (resolved, notes)。

    resolved = {"date": col|None, "amount": ..., "order_id": ..., "user_id": ..., "status": ..., "dimensions": [...]}
    notes    = 需要写进报告“口径说明”的提示列表
    """
    col_cfg = cfg["columns"]
    resolved = {"date": None, "amount": None, "order_id": None,
                "user_id": None, "status": None, "dimensions": []}
    notes = []

    resolved["date"] = _find_col(df, KEYWORDS["date"], col_cfg.get("date"))
    resolved["amount"] = _find_col(df, KEYWORDS["amount"], col_cfg.get("amount"))
    resolved["order_id"] = _find_col(df, KEYWORDS["order_id"], col_cfg.get("order_id"))
    resolved["user_id"] = _find_col(df, KEYWORDS["user_id"], col_cfg.get("user_id"))
    resolved["status"] = _find_col(df, KEYWORDS["status"], col_cfg.get("status"))

    # 维度列
    dims_cfg = col_cfg.get("dimensions") or []
    if dims_cfg:
        for d in dims_cfg:
            c = _find_col(df, [d], None)
            if c and c not in resolved["dimensions"]:
                resolved["dimensions"].append(c)
    else:
        for group in KEYWORDS["dimensions"]:
            c = _find_col(df, group, None)
            if c and c not in resolved["dimensions"]:
                resolved["dimensions"].append(c)
        resolved["dimensions"] = resolved["dimensions"][:4]

    # 类型校验
    if resolved["date"]:
        parsed = pd_to_datetime(df[resolved["date"]])
        ratio = parsed.notna().mean()
        if ratio < 0.8:
            notes.append(f"时间列「{resolved['date']}」可解析比例仅 {ratio:.0%}, 时间类检查已跳过")
            resolved["date"] = None
    if resolved["amount"]:
        nums = pd_to_numeric(df[resolved["amount"]])
        if nums.notna().mean() < 0.8:
            notes.append(f"金额列「{resolved['amount']}」可转数值比例仅 {nums.notna().mean():.0%}, 金额类检查已跳过")
            resolved["amount"] = None

    missing_roles = {"date": "时间", "amount": "金额", "order_id": "订单号",
                     "user_id": "用户ID", "status": "状态"}
    for role, cn in resolved.items():
        if role == "dimensions":
            continue
        if cn is None:
            notes.append(f"未识别到{missing_roles[role]}列, 相关检查已自动跳过(可在 config.json 中显式指定列名)")
    if not resolved["dimensions"]:
        notes.append("未识别到维度列(原因/渠道/星级等), 归因下钻将退化为整体量价分解")

    # 将解析列写回 cfg, 供后续检查使用
    cfg["_resolved"] = {k: v for k, v in resolved.items()}
    cfg["_notes"] = notes
    return resolved, notes


def pd_to_datetime(s):
    import pandas as pd
    return pd.to_datetime(s, errors="coerce")


def pd_to_numeric(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")
