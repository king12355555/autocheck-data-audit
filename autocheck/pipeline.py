# -*- coding: utf-8 -*-
"""主流程: 丢一张表 → 出报告。

run_analysis(df, config_path=None, use_llm=False, llm_cfg=None) -> dict
    {df, cols, sections, meta, facts, conclusion, report, llm_used, notes}
"""
import pandas as pd

from . import checks, config as cfg_mod, llm as llm_mod, report as report_mod


def _normalize(df, cols):
    """类型规整: 时间→datetime, 金额→numeric。"""
    df = df.copy()
    if cols.get("date"):
        df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    if cols.get("amount"):
        df[cols["amount"]] = pd.to_numeric(df[cols["amount"]], errors="coerce")
    return df


def run_analysis(df, config_path=None, use_llm=False, llm_cfg=None):
    cfg = cfg_mod.load_config(config_path)
    resolved, notes = cfg_mod.resolve_columns(df, cfg)
    df = _normalize(df, resolved)
    cfg["_resolved"] = resolved
    cfg["_notes"] = notes

    sections, meta = checks.run_all_checks(df, resolved, cfg)

    facts = report_mod.build_facts(meta, sections, cfg)
    llm_used = False
    conclusion = None
    if use_llm:
        conclusion = llm_mod.generate_conclusion(facts, llm_cfg)
        if conclusion:
            llm_used = True
    if not conclusion:
        conclusion = report_mod.render_template_conclusion(sections)

    report_md = report_mod.render_report(meta, sections, conclusion, notes,
                                         llm_used=llm_used, config=cfg)
    return {
        "df": df,
        "cols": resolved,
        "sections": sections,
        "meta": meta,
        "facts": facts,
        "conclusion": conclusion,
        "report": report_md,
        "llm_used": llm_used,
        "notes": notes,
    }
