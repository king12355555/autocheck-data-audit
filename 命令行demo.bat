@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [命令行演示] 用示例数据跑一遍体检, 报告保存为 示例报告.md
python main.py "sample_data\赔付明细_示例.csv" --out "示例报告.md"
echo.
echo 报告已生成: %~dp0示例报告.md
pause
