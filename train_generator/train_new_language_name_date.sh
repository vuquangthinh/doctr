USE_PYTORCH=1

MODEL=khm_name_date
# define VOCAB in VOCABS

python references/recognition/train_pytorch.py crnn_vgg16_bn --train_path ./data/${MODEL}/train_data --val_path ./data/${MODEL}/val_data --epochs 500  --input_size 32 --vocab khm2 --name ${MODEL}.pt
