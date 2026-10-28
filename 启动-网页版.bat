@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 AutoCheck 网页版...
echo 启动后浏览器会自动打开 http://localhost:8501
python -m streamlit run app.py
pause
