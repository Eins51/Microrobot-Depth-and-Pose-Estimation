export HF_ENDPOINT=https://hf-mirror.com/

#python train.py -c configs/pose_resnet50.yaml -d 3 --save results/exp5_pose_resnet50

python train_regression.py -c configs/depth_reg_resnet50.yaml -d 3 --save results/exp9_depth_regression_resnet50

