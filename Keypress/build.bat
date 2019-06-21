@echo off
C:/Users/smartin5/AppData/Local/Programs/Python/Python37/python.exe -m PyInstaller ^
    --noconfirm --log-level WARN ^
    --onefile --nowindow ^
    --name KeyTime ^
    cli.py