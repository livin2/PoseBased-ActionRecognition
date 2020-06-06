#  [PoseBased-ActionRecognition](https://github.com/livin2/PoseBased-ActionRecognition) 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

 An real-time action recognition application for webcam(RTMP/RTSP). https://github.com/livin2/PoseBased-ActionRecognition

## setup

### Requirements

- Linux
- Python 3.6.10+
- Cython
- Pytorch 1.1.0
- torchvision 0.3.0
- find (GNU findutils) 4.7.0
- viewnior 1.7
- AlphaPose

### Install Scripts

```
#install conda
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh
sudo bash Miniconda3-latest-Linux-x86_64.sh

#create env
conda create -n myenv python=3.6 -y
conda activate myenv

# install dependents
conda install pytorch==1.1.0 torchvision==0.3.0
conda install opencv
conda install jupyter notebook

#only for arch,please use apt for ubuntu
sudo pacman -Sy libyaml findutils viewnior npm

#pip essential 
#	numpy>=1.18.1
#	scipy>=1.3.2
#	Cython>=0.29.15
#	scikit-learn>=0.22.1
#	hiddenlayer>=0.3
#	loguru>=0.4.1
#	pandas>=1.0.1
#	Flask>=1.1.2
#   Flask-Cors>=3.0.8
#	Flask-SocketIO>=4.3.0
#	easydict>=1.9
#	tqdm>=4.42.0
#	PySnooper>=0.3.0
pip install -r requirement.txt

#AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git Alphapose
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
python Alphapose/setup.py build develop

#GUI dependencies
npm install -g @vue/cli
cd PoseUi
npm install
```

### Models

Download models from according to each `get_model.txt` in `\model` and make them look like:

```
.
├── act_dnnSingle_9
│   ├── epoch_1000.pth
│   └── epoch_500.pth
├── act_fcLstm_9
│   ├── epoch_1000.pth
│   └── epoch_500.pth
├── detector
│   ├── tracker
│   │   ├── jde.1088x608.uncertainty.pt
│   │   └── yolov3.cfg
│   └── yolo
│       ├── yolov3-spp.cfg
│       └── yolov3-spp.weights
├── pose_res152_duc
│   ├── 256x192_res152_lr1e-3_1x-duc.yaml
│   └── fast_421_res152_256x192.pth
├── pose_res50
│   ├── 256x192_res50_lr1e-3_1x.yaml
│   └── fast_res50_256x192.pth
└──  pose_res50_dcn
    ├── 256x192_res50_lr1e-3_2x-dcn.yaml
    └── fast_dcn_res50_256x192.pth
```

### Build GUI

```
cd PoseUi
npm run build
cd ..
cp -r PoseUI/dist/css static/
cp -r PoseUI/dist/js static/
cp PoseUI/dist/favicon.ico static/
cp PoseUI/dist/index.html templates/
```

modify `templates/index.html` , add `/static`  to all resources link,like:

```html
origin:
<link rel=icon href=/favicon.ico>
modified:
<link rel=icon href=/static/favicon.ico>
```

## Run

For all arguments detail, see`config/default_args`.

For minimal scripts to run, see `local_demo.py`. And simply run by `python local_demo.py` in conda env.

Run with GUI：

```
conda activate myenv
export FLASK_APP=flask_server.py 
flask run
```

## Train

models define in `actRec/models.py`

For train process and result,see [train/train-dnn.ipynb](train/train-dnn.ipynb) and [train/train-fclstm.ipynb](train/train-fclstm.ipynb).

Open and run them by `jupyter notebook` in conda env.

## License

```
Copyright 2019 github@livin2

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

[AlphaPose License]( https://github.com/MVIG-SJTU/AlphaPose/blob/master/LICENSE )