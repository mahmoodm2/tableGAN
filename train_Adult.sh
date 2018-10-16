#!/usr/bin/env bash
echo "Adult Data Sets"
python main.py --train --dataset=Adult --epoch=5 --test_id=OI_11_00
