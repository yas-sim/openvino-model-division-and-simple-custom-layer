#!/usr/bin/env bash

if [ ${INTEL_OPENVINO_DIR} = "" ] ; then
    source /opt/intel/openvino_2021/bin/setupvars.sh
fi

python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
  --saved_model_dir mnist-savedmodel \
  -b 1 \
  --model_name mnist \
  --output_dir models
