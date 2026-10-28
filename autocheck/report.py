# -*- coding: utf-8 -*-
"""事实清单组装 + 报告渲染。

- facts: 只包含代码算出来的事实(给 LLM 用, 防止它编数字)
- 模板报告: 全部由代码渲染, 数字必然真实
- LLM 结论: 只负责“把事实讲成人话”, 失败自动回退模板
"""
import datetime

import numpy as np
import pandas as pd

_LEVEL_CN = {"ok": "✅ 通过", "warn": "⚠️ 告警", "error": "🚨 异常"}


def _fmt_money(v):
    return f"{v:,.0f}"


# ---------------------------------------------------------------- 表格渲染
def _cell(v):
    if v is None:
        return ""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (np.integer, int)):
        return f"{int(v):,}"
    if isinstance(v, (np.floating, float)):
        if float(v).is_integer() and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:,.1f}"
    return str(v)


def _md_table(df, max_rows=20):
    if df is None or len(df) == 0:
        return "_无_"
    truncated = len(df) > max_rows
    total = len(df)
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if "行号" in str(c):  # 文件行号不用千分位, 方便对照原表
                cells.append(str(int(v)) if pd.notna(v) else "")
            else:
                cells.append(_cell(v))
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    if truncated:
        table += f"\n\n_上表共 {total} 行, 仅显示前 {max_rows} 行; 完整见同目录「问题数据明细.csv」_"
    return table


def build_issue_csv(sections):
    """汇总所有“问题行”为一张表(带问题类型+文件行号), 供导出 CSV/Excel。"""
    parts = []
    for s in sections:
        for f in (s.get("data", {}).get("frames") or []):
            if f["kind"] != "rows" or f["df"] is None or len(f["df"]) == 0:
                continue
            dd = f["df"].copy()
            dd.insert(0, "问题类型", f["title"].split("(")[0].strip())
            parts.append(dd)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    cols = ["问题类型", "文件行号"] + [c for c in df.columns if c not in ("问题类型", "文件行号")]
    return df[cols]


# ---------------------------------------------------------------- 大白话解读
def render_plain_summary(sections):
    """把检查结果翻译成“不用懂技术也看得懂”的几句话。"""
    d = {s["key"]: (s.get("data") or {}) for s in sections}
    counts = {k: v.get("counts", {}) for k, v in d.items()}
    out = []

    # 0. 红灯总数预览(防止只看到某几条“未发现”而产生一切正常的错觉)
    bad = [s for s in sections if s.get("status") in ("warn", "error") and s["key"] != "overview"]
    if bad:
        names = "、".join(s["title"].split("、", 1)[-1].rstrip("检查") for s in bad)
        out.append(f"⚠️ **本次体检共 {len(bad)} 个检查项亮红灯**({names}), 逐一说明如下。")
    else:
        out.append("✅ 本次体检所有检查项均通过, 未发现问题。")

    # 1. 数据本身的问题
    qc = counts.get("data_quality", {})
    parts = []
    if qc.get("negative"):
        parts.append(f"有 {qc['negative']} 条金额是**负数**(金额不该是负的, 可能是冲正或录入错误)")
    if qc.get("nan_amount"):
        parts.append(f"有 {qc['nan_amount']} 条金额是**空的**")
    if qc.get("outliers"):
        parts.append(f"有 {qc['outliers']} 条金额**特别大**, 远超正常水平")
    if qc.get("dup_rows"):
        parts.append(f"有 {qc['dup_rows']} 行**完全重复**的记录")
    if qc.get("blank_user"):
        parts.append(f"有 {qc['blank_user']} 条记录**没有用户ID**")
    if parts:
        out.append("**① 数据本身有问题**: " + "、".join(parts) + "。"
                   "→ 这类先找数据同学核对是不是录入/取数错误, 修好数据再做业务分析。")

    # 2. 时间波动 + 归因
    tr = counts.get("trend", {})
    at = counts.get("attribution", {})
    if tr.get("big_weeks"):
        s = f"**② 时间上有异常波动**: 有 {tr['big_weeks']} 个周的赔付金额环比变化超过 ±25%(正常波动一般到不了这么大)。"
        if at.get("has_peak"):
            delta = at.get("delta_money", 0)
            pct = at.get("delta_pct", 0) * 100
            s += f"最明显的是 **{at['peak_week']}** 那一周, 比前一周多赔了约 **{_fmt_money(delta)} 元(+{pct:.0f}%)**"
            if at.get("main_reason"):
                s += f", 增量主要来自**「{at['main_reason']}」"
            if at.get("main_channel"):
                s += f"、渠道「{at['main_channel']}」"
            if at.get("main_reason") or at.get("main_channel"):
                s += "**"
            s += "。→ 建议去查那一周对应的业务动作/供应商/规则有没有变化, 并核对峰期的几笔大额单(见下方明细)。"
        out.append(s)
    else:
        out.append("**② 时间趋势正常**: 各周赔付金额没有明显突变, 波动在正常范围内。")

    # 3. 重复/高频
    dp = counts.get("duplicates", {})
    bits = []
    if dp.get("dup_orders"):
        bits.append(f"{dp['dup_orders']} 个订单被赔付了不止一次(共多赔 {dp.get('dup_extra', 0)} 笔)")
    if dp.get("high_freq_users"):
        bits.append(f"{dp['high_freq_users']} 个用户在 30 天内申请了好几次")
    if bits:
        out.append("**③ 疑似重复赔付/刷单**: " + "、".join(bits) + "。"
                   "→ 先抽样人工核对是不是系统重复打款或有人恶意套利, 确认后再定规则。")
    else:
        out.append("**③ 未发现重复赔付或高频申请**。")

    # 4. 漏斗
    fu = counts.get("funnel", {})
    if fu.get("overall_rate") is not None:
        out.append(f"**④ 流程漏斗**: 从申请到最终打款, 整体通过率约 **{fu['overall_rate'] * 100:.1f}%**"
                   "。中间有流失的环节见下文第六节, 可进一步看是哪一步、为什么没走下去。")

    if not out:
        out.append("本次未发现明显问题。")
    return "\n".join(f"- {t}" for t in out)


# ---------------------------------------------------------------- 事实清单
def build_facts(meta, sections, cfg):
    lines = [f"表名: {cfg.get('table_name', '赔付明细')}"]
    lines.append(f"总记录数: {meta.get('rows', 0)} 行")
    if meta.get("date_min") is not None:
        lines.append(f"时间范围: {meta['date_min'].date()} ~ {meta['date_max'].date()}")
    if meta.get("amount_total") is not None:
        lines.append(f"赔付总额: {_fmt_money(meta['amount_total'])} 元, 单均: {_fmt_money(meta['amount_avg'])} 元")

    # 红灯总数(喂给大模型, 防止它漏看问题)
    n_bad = sum(1 for s in sections if s["key"] != "overview" and s.get("status") in ("warn", "error"))
    lines.append(f"检查结果总览: {len(sections) - 1} 个检查项中 {n_bad} 个存在告警/异常")
    if n_bad == 0:
        lines.append("注意: 所有检查项均通过, 可判定为无明显异常")

    for sec in sections:
        if sec["key"] == "overview":
            continue
        lines.append(f"\n[{sec['title']}] ({sec['status']})")
        for it in sec["items"]:
            lines.append(f"- [{it['level']}] {it['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 模板结论
def render_template_conclusion(sections):
    out = []
    errs = [it for s in sections for it in s["items"] if it["level"] in ("error", "warn")]
    if not errs:
        out.append("**总体判断**: 本期数据体检未发现明显异常, 各项指标处于正常波动区间, 建议保持周期性监控。")
        return "\n".join(out)

    out.append("**总体判断**: 发现若干需关注的问题, 按严重程度从高到低处理: ")
    errs = [it for s in sections for it in s["items"] if it["level"] == "error"]
    warns = [it for s in sections for it in s["items"] if it["level"] == "warn"]
    n = 1
    for it in (errs + warns)[:8]:
        out.append(f"{n}. {it['text']}")
        n += 1

    advice = []
    for s in sections:
        for a in s["advice"]:
            if a not in advice:
                advice.append(a)
    if advice:
        out.append("\n**下一步建议**:")
        for i, a in enumerate(advice[:6], 1):
            out.append(f"{i}. {a}")
    out.append("\n*说明: 以上结论由规则模板生成(未配置大模型 API Key)。配置 .env 后可获得 AI 生成的深度归因结论。*")
    return "\n".join(out)


# ---------------------------------------------------------------- 完整报告
def render_report(meta, sections, conclusion_md, notes, llm_used=False, config=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    table_cfg = (config or {}).get("checks", {})
    cfg_line = (f"突变阈值 ±{table_cfg.get('wow_threshold', 0.25) * 100:.0f}% · "
                f"动态阈值滚动 {table_cfg.get('rolling_days', 7)} 日均值+{table_cfg.get('zscore_k', 2.0):.0f}σ · "
                f"高频申请窗口 {table_cfg.get('dup_user_window_days', 30)} 天≥{table_cfg.get('dup_user_min_count', 3)} 次")

    lines = ["# 赔付数据体检报告", ""]
    lines.append(f"> 生成时间: {now}  |  记录数: {meta.get('rows', 0)} 行"
                 + (f"  |  时间范围: {meta['date_min'].date()} ~ {meta['date_max'].date()}" if meta.get("date_min") is not None else "")
                 + (f"  |  赔付总额: {_fmt_money(meta['amount_total'])} 元" if meta.get("amount_total") is not None else ""))
    lines.append("")
    lines.append("> 参数: " + cfg_line)
    lines.append("> 结论生成: " + ("大模型(基于确定性检查结果)" if llm_used else "规则模板(未配置 LLM, 可降级离线使用)"))
    lines.append("")

    # 大白话解读(放最前面)
    lines.append("## 💡 大白话解读(先看这个)")
    lines.append("")
    lines.append(render_plain_summary(sections))
    lines.append("")

    # 体检汇总
    lines.append("## 检查项汇总")
    lines.append("")
    lines.append("| 检查项 | 状态 | 摘要 |")
    lines.append("|---|---|---|")
    for s in sections:
        lines.append(f"| {s['title'].split('、', 1)[-1]} | {_LEVEL_CN.get(s['status'], s['status'])} | {s['summary']} |")
    lines.append("")

    for s in sections:
        lines.append(s["title"])
        lines.append("")
        for it in s["items"]:
            lines.append(f"- {it['text']}")
        lines.append("")

    lines.append("## 七、结论与建议")
    lines.append("")
    lines.append(conclusion_md.strip())
    lines.append("")

    # 问题数据明细(具体是哪几条记录)
    rows_frames = [f for s in sections for f in (s.get("data", {}).get("frames") or []) if f["kind"] == "rows"]
    rows_frames = [f for f in rows_frames if f["df"] is not None and len(f["df"]) > 0]
    lines.append("## 八、问题数据明细(直接看是哪几条数据有问题)")
    lines.append("")
    if rows_frames:
        for f in rows_frames:
            lines.append(f"### {f['title']}")
            lines.append("")
            lines.append(_md_table(f["df"]))
            lines.append("")
    else:
        lines.append("_本次未检出需要逐条查看的问题记录。_")
        lines.append("")

    # 附表
    tbl_frames = [f for s in sections for f in (s.get("data", {}).get("frames") or []) if f["kind"] == "table"]
    tbl_frames = [f for f in tbl_frames if f["df"] is not None and len(f["df"]) > 0]
    lines.append("## 九、附表(趋势 / 阈值 / 漏斗)")
    lines.append("")
    if tbl_frames:
        for f in tbl_frames:
            lines.append(f"### {f['title']}")
            lines.append("")
            lines.append(_md_table(f["df"]))
            lines.append("")
    else:
        lines.append("_无_")
        lines.append("")

    if notes:
        lines.append("## 附: 口径与说明")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("---")
    lines.append("*AutoCheck 赔付数据体检助手 · 数字均由确定性代码计算, 大模型仅基于事实生成文字结论*")
    return "\n".join(lines)
