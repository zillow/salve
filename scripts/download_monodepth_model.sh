

# Downloads from HoHoNet's model storage
# See https://drive.google.com/drive/folders/1BjPFidC9M3mBYshZjU5ZYAI1-TU_B2b6


HOHONET_CKPT_DIR=ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet
HOHONET_CKPT_FPATH=$HOHONET_CKPT_DIR/ep60.pth

mkdir -p $HOHONET_CKPT_DIR

GDRIVE_FILEID=1kZFPwdo36Uk7qP96yYUyQebZtGjsEabL
GDRIVE_URL='https://docs.google.com/uc?export=download&id='$GDRIVE_FILEID
wget --save-cookies cookies.txt $GDRIVE_URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $HOHONET_CKPT_FPATH $GDRIVE_URL'&confirm='$(<confirm.txt)


