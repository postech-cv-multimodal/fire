data_dir=/home/kimsubin/data/CUB_200_2011
src_idr=/home/kimsubin/workspace/fire

python train.py \
    --annot_dir $data_dir/annotations \
    --vision_dir $data_dir/images \
    --attributes_path $src_idr/playground/attribute_sample.txt \
    --checkpoint_name_or_path Salesforce/instructblip-flan-t5-xxl \
    --model_type vit \
    --gpuid 0 1 \
    --model_name vit_baseline \
    --max_epochs 10 \
    --batch_size 256 \
    --lr 5e-4