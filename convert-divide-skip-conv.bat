if "%INTEL_OPENVINO_DIR%" == "" (
    echo "OpenVINO environment variables are not set. Run setupvars.bat and then continue"
    exit
)

: Node names for '--input' and '--output' options are full-length node name in ** MO internal (intermediate) networkx graph ** (not a node name in TF)

: In this example, we divide the input model into 2 parts and intentionally skip 'target_conv_layer'.

python "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" ^
  --saved_model_dir mnist-savedmodel ^
  -b 1 ^
  --input conv2d_input ^
  --output StatefulPartitionedCall/sequential/max_pooling2d_1/MaxPool ^
  --model_name mnist_skip_1 ^
  --output_dir models

python "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" ^
  --saved_model_dir mnist-savedmodel ^
  -b 1 ^
  --input StatefulPartitionedCall/sequential/flatten/Reshape ^
  --output StatefulPartitionedCall/sequential/dense_1/Softmax ^
  --model_name mnist_skip_2 ^
  --output_dir models
