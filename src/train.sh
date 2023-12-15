data_dir=/home/kimsubin/data/CUB_200_2011
src_idr=/home/kimsubin/fire

CUDA_LAUNCH_BLOCKING=1 python train.py \
    --annot_dir $data_dir/annotations_2 \
    --vision_dir $data_dir/images \
    --attributes_path $src_idr/playground/attribute_sample.txt \
    --checkpoint_name_or_path Salesforce/instructblip-flan-t5-xxl \
    --model_type vit \
    --model_name vit \
    --max_epochs 100 \
    --batch_size 128 \
    --gpuid 0 1 \
    --seed 19 \
    --num_classes 100 \
    --embedding_size 768