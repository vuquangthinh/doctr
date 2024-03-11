USE_PYTORCH=1

MODEL=mrz_parseq
LANG=mrz_code
# define VOCAB in VOCABS

python references/recognition/train_pytorch.py parseq --lr 0.001 --font "/home/thanhpcc/Documents/testx/doctr/train_generator/OCRB Regular.ttf" --epochs 10 --pretrained  --input_size 32 --vocab ${LANG} --name ${MODEL} --early-stop # --resume ${MODEL}.pt