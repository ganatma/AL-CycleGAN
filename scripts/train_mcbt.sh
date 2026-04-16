set -ex
python train.py --dataroot ./MCBT --name mcbt_cyclegan --model cycle_gan --pool_size 50 --no_dropout
