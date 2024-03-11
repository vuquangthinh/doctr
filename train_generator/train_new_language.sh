USE_PYTORCH=1

MODEL=khm
# define VOCAB in VOCABS

# python references/recognition/train_pytorch.py crnn_vgg16_bn --lr 0.005 --train_path ./data/${MODEL}/train_data --val_path ./data/${MODEL}/val_data --epochs 30  --input_size 32 --vocab khm --resume crnn_vgg16_bn_20240307-011748.pt
python references/recognition/train_pytorch.py crnn_vgg16_bn --lr 0.001 --train_path ./data/${MODEL}/train_data --val_path ./data/${MODEL}/val_data --epochs 500  --input_size 32 --vocab khm --resume crnn_vgg16_bn_20240308-145243.pt
