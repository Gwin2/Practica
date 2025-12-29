@echo off
echo Запуск полного конвейера ViT детекции дронов...
echo.

REM Активация виртуального окружения (если есть)
call venv\Scripts\activate.bat

REM Запуск конвейера
python complete_vit_drone_detection_pipeline.py ^
  --data "C:\Users\snytk\PycharmProjects\YOLOTrain\DatasetFromMAI\data.yaml" ^
  --test-video "C:\Users\snytk\PycharmProjects\YOLOTrain\videoMAI\01-02.mp4" ^
  --output-dir "C:\Users\snytk\PycharmProjects\YOLOTrain\ViT_Results"

pause