if "%INTEL_OPENVINO_DIR%" == "" (
    echo "OpenVINO environment variables are not set. Run setupvars.bat and then continue"
    exit
)
python "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" ^
  --saved_model_dir mnist-savedmodel ^
  -b 1 ^
  --model_name mnist ^
  --output_dir models
