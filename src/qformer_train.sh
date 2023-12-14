blip_model_file=/home/jhkim980112/workspace/code/CV_project/models
blip_processor_file=/home/jhkim980112/workspace/code/CV_project/processors
image_path=/home/jhkim980112/workspace/dataset/CUB_200_2011/CUB_200_2011/images
train_data_path=/home/jhkim980112/workspace/dataset/CUB_200_2011/annotations/train.json
attribute_path=/home/jhkim980112/workspace/code/CV_project/fire/playground/attribute_sample.txt
checkpoint_name_or_path=/home/jhkim980112/workspace/code/CV_project/models/

python qformer_train.py \
    --gpu 3 \
    --batch_size 30 \
    --blip_model_file $blip_model_file \
    --blip_processor_file $blip_processor_file \
    --image_path $image_path \
    --train_data_path $train_data_path \
    --attribute_path $attribute_path \
    --checkpoint_name_or_path $checkpoint_name_or_path \