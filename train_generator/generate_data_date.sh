MODEL=khm_date
LANG=km
TOTAL_IMAGES=10000
WORDS_PER_IMAGE=1

####

TOTAL_VAL_IMAGES=$((TOTAL_IMAGES * 10 / 50))
prefix=./train_generator
DATASET_FOLDER=../data/${MODEL}

cd ${prefix}

mkdir -p ${DATASET_FOLDER}/train_data/images

echo "Generating train_data for ${MODEL}"
trdg -l ${LANG} --font_dir ./fonts --font Battambang-Regular -c ${TOTAL_IMAGES} \
  -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -na 2 -f 32 -bl 3 -rbl \
  --dict ./${MODEL}.txt -t 64 \
  --word_split --output_dir ${DATASET_FOLDER}/train_data/images
python ./transform_dataset.py --input_file ${DATASET_FOLDER}/train_data/images/labels.txt --output_file ${DATASET_FOLDER}/train_data/labels.json

echo "Generating val_data for ${MODEL}"
trdg -l ${LANG} --font_dir ./fonts -k 1 -rk -c ${TOTAL_VAL_IMAGES} -bl 3 -rbl -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -b 1 -bl 1 -rbl -na 2 -f 32 --dict ./${MODEL}.txt -t 64 --word_split --output_dir ${DATASET_FOLDER}/val_data/images
python ./transform_dataset.py --input_file ${DATASET_FOLDER}/val_data/images/labels.txt --output_file ${DATASET_FOLDER}/val_data/labels.json





echo "Train for ${MODEL}"
cd ..
python references/recognition/train_pytorch.py crnn_vgg16_bn --train_path ./data/${MODEL}/train_data \
  --val_path ./data/${MODEL}/val_data --epochs 200 --input_size 32 --vocab khm --name khm_date --pretrained
