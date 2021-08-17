#!/usr/bin/env bash

if [ ${INTEL_OPENVINO_DIR} = "" ] ; then
    source /opt/intel/openvino_2021/bin/setupvars.sh
fi


# Node names for '--input' and '--output' options are full-length node name in ** MO internal (intermediate) networkx graph ** (not a node name in TF)

python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
  --saved_model_dir mnist-savedmodel \
  -b 1 \
  --input conv2d_input \
  --output StatefulPartitionedCall/sequential/target_conv_layer/Relu \
  --model_name mnist_div_1 \
  --output_dir models

python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
  --saved_model_dir mnist-savedmodel \
  -b 1 \
  --input StatefulPartitionedCall/sequential/flatten/Reshape \
  --output StatefulPartitionedCall/sequential/dense_1/Softmax \
  --model_name mnist_div_2 \
  --output_dir models
