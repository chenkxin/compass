#!/bin/bash
## INSTRUCTIONS
# This batch file is used for train compass in the ModelNet dataset. It is based on https://github.com/charlesq34/pointnet
#
# First download ModelNet dataset from: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
# For more information about this dataset please refer to: https://github.com/charlesq34/pointnet
#
# Create files named train.csv and test.csv
# We adopt files ply_data_train[0:4].h5 for train and ply_data_test[0:1].h5 for test
#
# Train your own network by using train_compass_ModelNet.sh or download te pretrained weights
#
# If you find this work useful please cite

#data_path="/media/mmarcon/data/DATASETS/modelnet40_ply_hdf5_2048/"
data_path="/home/chenkangxin/compass/data/modelnet40" #path to the dataset
n_epocs=150
dataset_type="modelnet40" # or shapenet
nm_train="classification_compass"
date_time=`date +%Y-%m-%d_%H-%M`
#output_folder="/media/mmarcon/data/Train_SOUND/PointNet/$nm_train-$date_time/"
comment="NRwithcompass-AR"
output_folder="/home/chenkangxin/compass/log/Train_SOUND/PointNet/$nm_train-$date_time-$comment"
feature_transform=0 #use feature t-net on PointNet architecture
AR=1 #use arbitrary rotations
rotate_axis='all' #define augmentation mode
debug=0

# set path for the compass pre-trained weights
#checkpoint=25200
checkpoint=0
#train_name="2020-05-26_09-39-Train_btw24_near_sonic_3laySO3_ModelNet_no_augmentation"
train_name="2022-03-21_14-42-Compass_ModelNet"
#path_lrf="/media/mmarcon/data/Train_SOUND/ModelNet/$train_name/checkpoints/lrf_layer_$checkpoint.pkl" #path to lrf pre-trained weights
#path_lrf="/home/chenkangxin/compass/log/Train_SOUND/ModelNet/$train_name/checkpoints/lrf_layer_$checkpoint.pkl" #path to lrf pre-trained weights
#path_s2="/media/mmarcon/data/Train_SOUND/ModelNet/$train_name/checkpoints/s2_layer_$checkpoint.pkl" #path to s2 pre-trained weights
#path_s2="/home/chenkangxin/compass/log/Train_SOUND/ModelNet/$train_name/checkpoints/s2_layer_$checkpoint.pkl" #path to s2 pre-trained weights

# use pretrained compass from official
path_lrf="/home/chenkangxin/compass/pretrained_models/modelnet40/lrf_layer.pkl"
path_s2="/home/chenkangxin/compass/pretrained_models/modelnet40/s2_layer.pkl"
# To enable test mode, define the pre-trained classifier weights path and set --path_pointnet parameter
#path_pointnet="/media/mmarcon/data/Train_SOUND/PointNet/Train_classification_PointNet_plain_with_no_augmented_point_on_feature_off_plain-2020-08-12_11-42/cls_model_149.pth"
path_pointnet="/home/chenkangxin/compass/log/Train_SOUND/PointNet/classification_compass-2022-03-24_21-51-NR-AR/cls_model_149.pth"

file_train="$data_path/train.csv"
file_test="$data_path/test.csv"

#export PYTHONPATH="/home/mmarcon/workspace_python/compass"
export PYTHONPATH="/home/chenkangxin/compass"

#python ../apps/train_classification_pointnet.py --dataset $data_path --nepoch=$n_epocs --dataset_type $dataset_type \
#                                                --outf $output_folder --batchSize 50 --num_points 2048 \
#                                                --workers 8 --feature_transform $feature_transform \
#                                                --rotate_axis $rotate_axis --debug $debug --arbitrary_rotations $AR \
#                                                --path_ckp_layer_lrf $path_lrf --path_ckp_layer_s2 $path_s2 \
#                                                --file_list_train $file_train --file_list_test $file_test
#                                                #--path_pointnet $path_pointnet

python ./apps/train_classification_pointnet.py --dataset $data_path --nepoch $n_epocs --dataset_type $dataset_type \
                                                --outf $output_folder --batchSize 50 --num_points 2048 \
                                                --workers 8 --feature_transform $feature_transform \
                                                --rotate_axis $rotate_axis --debug $debug --arbitrary_rotations $AR \
                                                --path_ckp_layer_lrf $path_lrf --path_ckp_layer_s2 $path_s2 \
                                                --file_list_train $file_train --file_list_test $file_test \
#                                                --path_pointnet $path_pointnet
