nnodes=8
master_addr="100.120.249.133"

python3 -m torch.distributed.launch --master_port 2345 --nproc_per_node=8 \
            --nnodes=${nnodes} --node_rank=$1  \
            --master_addr=${master_addr} \
            031-5.InstaHide_vitb16_resume_ckpt_multinodes.py --batch_size 1024 --lr 2e-1 --lr_min 4e-3 --k 2  --is_attack True --attacker_dataset imdb
# bs, lr, lr_min *= {nnodes}
# 1st node command: 
# sh 031-5.nnodes.sh 0
# 2nd node command:
# sh 031-5.nnodes.sh 1