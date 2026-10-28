# -*- coding: utf-8 -*-
"""命令行入口: python main.py 你的表.csv [--config config.json] [--llm] [--out 报告.md]"""
import argparse
import os
import sys

from autocheck.loader import read_table
from autocheck.pipeline import run_analysis
from autocheck.report import build_issue_csv


def main():
    ap = argparse.ArgumentParser(description="AutoCheck 赔付数据体检助手(命令行版)")
    ap.add_argument("input", help="数据文件路径 (csv/tsv/txt/xlsx)")
    ap.add_argument("--config", default=None, help="config.json 路径(可选)")
    ap.add_argument("--llm", action="store_true", help="启用大模型结论(需 .env 配置 API Key)")
    ap.add_argument("--out", default=None, help="报告输出路径(默认打印到屏幕)")
    ap.add_argument("--quiet", action="store_true", help="不打印报告正文")
    args = ap.parse_args()

    print(f"[1/3] 读取数据: {args.input}")
    df = read_table(args.input)
    print(f"      共 {len(df)} 行 × {len(df.columns)} 列")

    print("[2/3] 运行确定性检查 ...")
    result = run_analysis(df, config_path=args.config, use_llm=args.llm)
    print(f"      结论生成: {'大模型' if result['llm_used'] else '规则模板(未配置 LLM)'}")

    print("[3/3] 渲染报告 ...")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result["report"])
        print(f"      报告已保存: {os.path.abspath(args.out)}")

    # 导出问题数据明细(带文件行号, Excel 可直接打开)
    issues_df = build_issue_csv(result["sections"])
    if issues_df is not None:
        base = os.path.dirname(os.path.abspath(args.out)) if args.out else os.getcwd()
        ipath = os.path.join(base, "问题数据明细.csv")
        issues_df.to_csv(ipath, index=False, encoding="utf-8-sig")
        print(f"      问题数据明细已导出: {os.path.abspath(ipath)} (共 {len(issues_df)} 条问题记录)")
        if not args.quiet:
            print("      问题行速查(文件行号=表头第1行, 数据从第2行起):")
            for s in result["sections"]:
                for f in (s.get("data", {}).get("frames") or []):
                    if f["kind"] == "rows" and f["df"] is not None and len(f["df"]):
                        nums = [str(int(x)) for x in f["df"]["文件行号"]][:8]
                        more = " …" if len(f["df"]) > 8 else ""
                        total = len(f["df"])
                        print(f"        - {f['title'].split('(')[0].strip()}: 文件行 {', '.join(nums)}{more} 等共 {total} 条")
    else:
        print("      未检出需要逐条查看的问题记录")
    if not args.quiet:
        print("\n" + "=" * 60)
        print(result["report"])
        print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
