# DTA 코드 

## 환경

cycler==0.10.0

kiwisolver==1.1.0

matplotlib==3.1.1

numpy==1.17.0

Pillow==6.2.0

protobuf==3.9.1

pyparsing==2.4.2

python-dateutil==2.8.0

six==1.12.0

tensorboardX==1.8

torch==1.2.0

torchvision==0.4.0

tqdm==4.34.0

python main.py --config_path ./configs/resnet50_dta_vat.json


## dataset code 종류
visdasource

visdatarget

amazon

dslr

webcam


## transforms type source  종류
visda_standard_source

visda_standard

amazon_source

amazon_source

dslr_source

webcam_source


## transforms type target  종류

visda_standard_target

amazon_target

dslr_target

webcam_target

## transforms type test 종류

visda_standard

amazon_test

dslr_test

webcam_test