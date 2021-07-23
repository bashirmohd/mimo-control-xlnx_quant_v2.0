#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT_DIR=$(echo $THIS_DIR | sed 's/fpga\/xlnx_dqn//')
TRAIN_DIR=$REPO_ROOT_DIR/feedback
TRAIN_RES_DIR=$TRAIN_DIR/nn_trained
TRAIN_LOCAL_DIR=$THIS_DIR/nn_trained
DSET=$TRAIN_LOCAL_DIR/3by3_30deg_dat.pt
SAMPLE=$TRAIN_LOCAL_DIR/sample*
WEIGHT=$TRAIN_LOCAL_DIR/3by3_30deg.pth
CONFIG=$TRAIN_LOCAL_DIR/3by3_30deg_config.json
TARGET_DIR=$THIS_DIR/target_zcu102

rm -rf __pycache__
rm -rf quant_model
rm -rf compiled_model
rm -rf target_*

python3 -m pip install --upgrade pip && python3 -m pip install gym  && python3 -m pip install stable-baselines3 

# run training
# deprecated as we use the mimo-control training
# python -u train.py 2>&1 | tee train.log

# real training
if [[ "$1" == "--train" ]]; then
  rm -rf $TRAIN_LOCAL_DIR
  cd $TRAIN_DIR
  make clean && make all && cp -r $TRAIN_RES_DIR $THIS_DIR
fi

cd $THIS_DIR

# quantize & export quantized model
python3 -u quantize.py --dset $DSET --weight $WEIGHT --config $CONFIG --quant_mode calib 2>&1 | tee quant_calib.log
python3 -u quantize.py --dset $DSET --weight $WEIGHT --config $CONFIG --quant_mode test  2>&1 | tee quant_test.log

echo ""
echo ""
echo "-------------------------------"
echo "Python-based Quantization Done!"
echo "-------------------------------"
echo ""
echo ""


# compile for target board
bash compile.sh zcu102

# make target folder
python3 -u target.py --target zcu102

# copy dataset and samples into target directory
cp $DSET $TARGET_DIR
cp $SAMPLE $TARGET_DIR

echo ""
echo ""
echo "-----------------------------------------"
echo "Done Quantizing and Compiling!"
echo "You can find the target board results in:"
echo "$TARGET_DIR"
echo "-----------------------------------------"
echo ""
echo ""