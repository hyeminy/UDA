### Requirements
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

DTA에서는 OPTOIN 값이 많다 JSON 파일에서 넘기자

DTA 코드의 단점은 dataset 별로 class를 생성해야 한다
datasets 폴더안에 dataset.py를 생성해야 하고, init안에 dataset code안에 새로 생성한 class의 이름을 추가해야 한다
진짜 별로다...


### transform_type 종류

'visda_standard_source'

'visda_standard_target' 

'amazon_source'

'webcam_source'

'dslr_source'

'amazon_target'

'webcam_target'

'dslr_target'

### transform_type_test 종류

'visda_standard'

'amazon_test'

'dslr_test'

'webcam_test'

### source_dataset_code, target_dataset_code 종류


'visdasource'

'visdatarget'

'amazon'

'webcam'

'dslr'