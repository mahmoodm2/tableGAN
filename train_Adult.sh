#!/usr/bin/env bash
echo "Adult Data Sets"
python main.py --train --dataset=Adult --epoch=5 --test_id=Origin
