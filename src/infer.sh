data_dir=/home/kimsubin/data/CUB_200_2011
src_dir=/home/kimsubin/workspace/fire
model_dir=$src_dir/src/model_ckpt

CUDA_VISIBLE_DEVICES=3 python infer.py \
    --annot_dir $data_dir/annotations \
    --vision_dir $data_dir/images \
    --attributes_path $src_dir/playground/attribute_sample.txt \
    --checkpoint_name_or_path $model_dir"/model_-last.ckpt" \
    --model_type vit \
    --backbone Salesforce/instructblip-flan-t5-xxl \
