MODEL=khm_name
LANG=km
TOTAL_IMAGES=12000
WORDS_PER_IMAGE=1

####

TOTAL_VAL_IMAGES=$((TOTAL_IMAGES * 20 / 50))
prefix=./train_generator

DATASET_FOLDER=../data/${MODEL}

cd ${prefix}

# mkdir -p ${DATASET_FOLDER}/train_data/images

# echo "Generating train_data for ${MODEL}"
# trdg -l ${LANG} --font_dir /home/thanhpcc/Documents/testx/doctr/train_generator/moul-font --font "Moul Regular" -c ${TOTAL_IMAGES} -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -rbl -bl 2 -na 2 -f 32 --dict ./${MODEL}.txt -t 64 --word_split --output_dir ${DATASET_FOLDER}/train_data/images
# python ./transform_dataset.py --input_file ${DATASET_FOLDER}/train_data/images/labels.txt --output_file ${DATASET_FOLDER}/train_data/labels.json


# echo "Generating val_data for ${MODEL}"
# trdg -l ${LANG} --font_dir ./moul-font -k 1 -rk -c ${TOTAL_VAL_IMAGES} -tc '#000000,#AAAAAA' -w ${WORDS_PER_IMAGE} -b 1 -bl 2 -rbl -na 2 -f 32 --dict ./${MODEL}.txt -t 64 --word_split --output_dir ${DATASET_FOLDER}/val_data/images
# python ./transform_dataset.py --input_file ${DATASET_FOLDER}/val_data/images/labels.txt --output_file ${DATASET_FOLDER}/val_data/labels.json

cd ..

# python references/recognition/train_pytorch.py crnn_vgg16_bn --lr 0.001 --train_path ./data/${MODEL}/train_data --val_path ./data/${MODEL}/val_data --epochs 1000  --input_size 32 --vocab khm --name ${MODEL} --resume ${MODEL}.pt
python references/recognition/train_pytorch.py crnn_vgg16_bn --lr 0.001 --train_path ./data/${MODEL}_manual/train_data --val_path ./data/${MODEL}_manual/val_data --epochs 5  --input_size 32 --vocab khm --name ${MODEL} --resume ${MODEL}.pt