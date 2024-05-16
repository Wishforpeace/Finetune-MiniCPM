#!/bin/bash
python /home/mnt/wyx/src/Finetune-MiniCPM/finetune/test_cosmosqa_ddp.py 0 &
python /home/mnt/wyx/src/Finetune-MiniCPM/finetune/test_cosmosqa_ddp.py 1 &
wait

