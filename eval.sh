# 定义主机名变量
expected_hostname="autodl-container-1e1b4d8ddb-a7794cf7"

# 获取当前主机名
current_hostname=$(hostname)

# 比较当前主机名与期望的主机名
if [ "$current_hostname" == "$expected_hostname" ]; then
    python eval.py --data_dir=/root/autodl-tmp/dataset/polyp/test/ --checkpoint=/applications/graduate_design/model/finetuned/SSFam/best.pth --save_dir=/applications/graduate_design/RGB_SSFAM/result --dataset=divide
else
    python train.py --train_data_dir=/applications/graduate_design/dataset/polyp/train --val_data_dir=/applications/graduate_design/dataset/polyp/test --checkpoint=/applications/graduate_design/model/init/sam_vit_l_0b3195.pth --save_path=/applications/graduate_design/polyp__SAM/saved/ --epoch=100 --val_dataset=overall --lr=5e-5 --batchsize=2
fi