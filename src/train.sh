data_dir=/home/kimsubin/data/CUB_200_2011
src_idr=/home/kimsubin/workspace/fire

CUDA_LAUNCH_BLOCKING=1 python train.py \
    --annot_dir $data_dir/annotations \
    --vision_dir $data_dir/images \
    --attributes_path $src_idr/playground/attribute_sample.txt \
    --checkpoint_name_or_path Salesforce/instructblip-flan-t5-xxl \
    --model_type vit \
    --gpuid 1 3 \
    --model_name vit \
    --max_epochs 30 \
    --batch_size 256 \
    --seed 42