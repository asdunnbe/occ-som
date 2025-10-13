#!/usr/bin/env bash

echo "backppack"
echo

python run_training.py \
  --work-dir OUTPUT-01/backpack --port 6010  data:iphone --data.data-dir data/backpack

echo
echo "v4"
echo

python run_training.py \
  --work-dir OUTPUT-01/v4 --port 6010  data:custom --data.data-dir data/c3vd/v4
