# -*- coding: utf-8 -*-
"""网页版入口: streamlit run app.py
丢一张表进去 → 自动体检 → 出报告(可下载)。
"""
import os

import pandas as pd
import streamlit as st

from autocheck.llm import llm_settings
from autocheck.loader import read_table
from autocheck.pipeline import run_analysis
from autocheck.report import build_issue_csv, render_plain_summary

st.set_page_config(page_title="AutoCheck 赔付数据体检助手", page_icon="🔍", layout="wide")
st.title("🔍 AutoCheck 赔付数据体检助手")
st.caption("丢一张表(CSV / Excel / SQL 导出)进来, 自动检查数据质量、波动归因、阈值预警、重复赔付, 并生成分析报告。"
           "**数字全部由代码计算, 大模型只负责写结论**, 不会瞎编数字。")

with st.sidebar:
    st.header("⚙️ 设置")
    mode = st.radio("结论生成方式", ["规则模板(离线可用)", "大模型(需 API Key)"], index=0)
    use_llm = mode.startswith("大模型")
    if use_llm:
        env = llm_settings()
        base_url = st.text_input("Base URL", value=env["base_url"])
        api_key = st.text_input("API Key", value=env["api_key"], type="password",
                                help="可留空: 读 .env 或在此填写(仅本次会话使用)")
        model = st.text_input("模型", value=env["model"])
        if not api_key:
            st.warning("未填 Key, 将自动降级为规则模板")
        llm_cfg = {"base_url": base_url, "api_key": api_key, "model": model}
    else:
        llm_cfg = None

    st.divider()
    st.caption("高级参数(可选覆盖)")
    wow_thr = st.slider("周环比突变阈值 %", 10, 100, 25, 5)
    roll_days = st.slider("动态阈值滚动窗口(天)", 3, 30, 7)
    z_k = st.slider("动态阈值: 均值 + K 倍标准差", 1.0, 4.0, 2.0, 0.5)
    cfg_override = {
        "checks": {
            "wow_threshold": wow_thr / 100,
            "rolling_days": roll_days,
            "zscore_k": z_k,
        }
    }

st.divider()

col_l, col_r = st.columns([1, 2])
with col_l:
    up = st.file_uploader("上传数据表", type=["csv", "tsv", "txt", "xlsx"], help="支持中文表头, 列名自动识别")
    cfg_file = st.file_uploader("(可选)config.json", type=["json"])
    run_btn = st.button("🚀 开始体检", type="primary", use_container_width=True)

if run_btn and up is None:
    st.warning("请先上传数据表")
    run_btn = False

if run_btn and up is not None:
    with st.spinner("读取数据 ..."):
        try:
            df = read_table(up)
            st.session_state["df"] = df
        except Exception as e:
            st.error(f"读取失败: {e}")
            st.stop()
    with st.spinner("运行确定性检查 + 生成报告 ..."):
        try:
            config_path = None
            if cfg_file is not None:
                tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_user_config.json")
                with open(tmp, "wb") as f:
                    f.write(cfg_file.getvalue())
                config_path = tmp
            result = run_analysis(df, config_path=config_path, use_llm=use_llm, llm_cfg=llm_cfg)
            st.session_state["result"] = result
            if config_path and os.path.exists(config_path):
                os.remove(config_path)
        except Exception as e:
            st.error(f"分析失败: {e}")
            st.stop()

if "result" in st.session_state:
    result = st.session_state["result"]
    meta = result["meta"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("记录数", f"{meta.get('rows', 0):,}")
    if meta.get("amount_total") is not None:
        c2.metric("赔付总额(元)", f"{meta['amount_total']:,.0f}")
        c3.metric("单均赔付(元)", f"{meta['amount_avg']:,.0f}")
    if meta.get("date_min") is not None:
        c4.metric("时间范围", f"{meta['date_min'].date()} ~ {meta['date_max'].date()}")

    st.markdown("---")
    st.markdown("### 💡 大白话解读")
    st.markdown(render_plain_summary(result["sections"]))

    rows_frames = [f for s in result["sections"]
                   for f in (s.get("data", {}).get("frames") or [])
                   if f["kind"] == "rows" and f["df"] is not None and len(f["df"]) > 0]
    tbl_frames = [f for s in result["sections"]
                  for f in (s.get("data", {}).get("frames") or [])
                  if f["kind"] == "table" and f["df"] is not None and len(f["df"]) > 0]

    st.markdown("---")
    st.markdown("### 🚨 哪几行数据有问题")
    st.caption("「文件行号」就是你的表格里的行号(表头算第 1 行, 第一条数据在第 2 行)。可点列头排序, 或下载 CSV 用 Excel 筛选。")
    if rows_frames:
        issues_df = build_issue_csv(result["sections"])
        if issues_df is not None:
            st.download_button("⬇️ 下载「问题数据明细.csv」(Excel 可直接打开)",
                               issues_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="问题数据明细.csv", mime="text/csv", type="primary")
        for f in rows_frames:
            st.markdown(f"**{f['title']}**")
            st.dataframe(f["df"], use_container_width=True, hide_index=True)
    else:
        st.info("本次未检出需要逐条查看的问题记录")

    st.markdown("---")
    st.markdown("### 📋 完整体检报告(各检查项细节与建议)")
    st.markdown(result["report"])
    st.download_button("⬇️ 下载 Markdown 报告", result["report"], file_name="赔付数据体检报告.md", mime="text/markdown")

    with st.expander("📊 附表(每周趋势 / 阈值日期 / 漏斗)"):
        if tbl_frames:
            for f in tbl_frames:
                st.markdown(f"**{f['title']}**")
                st.dataframe(f["df"], use_container_width=True, hide_index=True)
        else:
            st.info("无附表")

    with st.expander("🔎 查看原始数据预览(前 1000 行)"):
        st.dataframe(st.session_state["df"].head(1000), use_container_width=True)

    with st.expander("🧾 事实清单(喂给大模型的内容, 全部来自确定性计算)"):
        st.text(result["facts"])
