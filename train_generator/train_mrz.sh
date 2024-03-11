
LANG=en
TOTAL_IMAGES=20000
WORDS_PER_IMAGE=1
MODEL=mrz
DATASET_FOLDER=../data/${MODEL}
TOTAL_VAL_IMAGES=$((TOTAL_IMAGES * 20 / 50))

# python references/recognition/train_pytorch.py crnn_vgg16_bn --lr 0.001 --train_path ./data/${MODEL}/train_data --val_path ./data/${MODEL}/val_data --epochs 500  --input_size 32 --vocab english --name mrz_code


cd ./train_generator


# echo "Generating train_data for ${MODEL}"
# trdg -l ${LANG} --font_dir ./ --font "OCRB Regular"  -c ${TOTAL_IMAGES} -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -b 1 -na 2 -f 32 -bl 2 --dict ./${MODEL}.txt -t 64 --word_split --output_dir ${DATASET_FOLDER}/train_data/images
# python ./transform_dataset.py --input_file ${DATASET_FOLDER}/train_data/images/labels.txt --output_file ${DATASET_FOLDER}/train_data/labels.json


# echo "Generating val_data for ${MODEL}"
# trdg -l ${LANG} --font_dir ./ --font "OCRB Regular" -c ${TOTAL_VAL_IMAGES} -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -b 1 -bl 2 -rbl -na 2 -f 32 --dict ./${MODEL}.txt -t 64 --word_split --output_dir ${DATASET_FOLDER}/val_data/images
# python ./transform_dataset.py --input_file ${DATASET_FOLDER}/val_data/images/labels.txt --output_file ${DATASET_FOLDER}/val_data/labels.json


echo "Train for ${MODEL}"
cd ..
python references/recognition/train_pytorch.py crnn_vgg16_bn --train_path ./data/${MODEL}/train_data \
  --val_path ./data/${MODEL}/val_data --epochs 100 --input_size 32 --vocab latin --name mrz_code --pretrained --resume mrz_code.pt
