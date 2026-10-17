@echo off
title Spark-Decide Setup Downloader
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Download_SparkDecide.ps1"
if %ERRORLEVEL% NEQ 0 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/maxpy9/spark-decide/main/Download_SparkDecide.ps1 | iex"
)
