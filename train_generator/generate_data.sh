MODEL=khm
LANG=km
TOTAL_IMAGES=200000
WORDS_PER_IMAGE=1

####

TOTAL_VAL_IMAGES=$((TOTAL_IMAGES * 10 / 50))
prefix=./train_generator
DATASET_FOLDER=../data/${MODEL}

cd ${prefix}

mkdir -p ${DATASET_FOLDER}/train_data/images

echo "Generating train_data for ${MODEL}"
trdg -l ${LANG} --font_dir ./fonts -c ${TOTAL_IMAGES} -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -na 2 -f 32 --dict ./${MODEL}.txt -t 64 --word_split --output_dir ${DATASET_FOLDER}/train_data/images
python ./transform_dataset.py --input_file ${DATASET_FOLDER}/train_data/images/labels.txt --output_file ${DATASET_FOLDER}/train_data/labels.json


echo "Generating val_data for ${MODEL}"
trdg -l ${LANG} --font_dir ./fonts -k 1 -rk -c ${TOTAL_VAL_IMAGES} -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -b 1 -bl 1 -rbl -na 2 -f 32 --dict ./${MODEL}.txt -t 64 --word_split --output_dir ${DATASET_FOLDER}/val_data/images
python ./transform_dataset.py --input_file ${DATASET_FOLDER}/val_data/images/labels.txt --output_file ${DATASET_FOLDER}/val_data/labels.json
