@echo off
echo Installing dependencies...
py -m pip install -r requirements.txt --quiet
echo.

echo Starting Stock Market Chatbot...
echo.
py -m streamlit run smc_0_4_release.py
pause
