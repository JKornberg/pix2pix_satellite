set -ex
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
python train.py --dataroot ./datasets/landsat_4 --name landsat_4p2p --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gan_mode wgangp --lr .0002
