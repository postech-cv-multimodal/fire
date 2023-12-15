data_dir=/home/kimsubin/data/CUB_200_2011
src_dir=/home/kimsubin/fire

python infer.py \
    --annot_dir $data_dir/annotations \
    --vision_dir $data_dir/images \
    --attributes_path $src_dir/playground/attribute_sample_question.txt \
    --model_type vit+qformer \
    --backbone Salesforce/instructblip-flan-t5-xxl \
    --num_classes 100 \
    --embedding_size 768 \
    --num_attributes 9
