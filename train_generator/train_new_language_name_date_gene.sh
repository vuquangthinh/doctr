USE_PYTORCH=1

MODEL=khm_name2
# define VOCAB in VOCABS

python references/recognition/train_pytorch.py crnn_vgg16_bn --lr 0.005 --font /home/thanhpcc/Documents/testx/doctr/train_generator/moul-font/Moul-Regular.ttf --train-samples 10000 --val-samples 2000 --epochs 1000 --input_size 32 --vocab khm --name ${MODEL} --resume ${MODEL}.pt --early-stop