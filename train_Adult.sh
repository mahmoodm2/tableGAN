#!/usr/bin/env bash
echo "Adult Data Sets"
python main.py --train --dataset=Adult --epoch=200 --test_id=dcgan
