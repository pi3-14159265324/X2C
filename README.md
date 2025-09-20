# Human-to-humanoid facial expression imitation 🤖 
This repository is the official implementation of the imitation framework **X2CNet** in paper 

_**X2C: A Dataset for Realistic Human-to-Humanoid Facial Expression Imitation**_ 

![Alt text](docs/static/images/imitation_framework2.png)


## 🚀 Getting Started 
🔧 **Clone the Code and Set Up the Environment**

```bash
git clone git@github.com:pi3-14159265324/X2C.git
cd X2C

# create env using conda
conda create -n x2c python=3.9
conda activate x2c
# for cuda 12.1
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

 📦 **Install Python Dependencies**

```setup
pip install -r requirements.txt
```


## 🛠️ Dataset Preprocessing

We provide a preprocessing script to correct image paths after downloading the X2C dataset (available soon).
You can find it here: [`misc/dataset_preprocessing.py`](misc/dataset_preprocessing.py)

**How to Use**

coming soon

 **Make sure to replace** /path/to/X2C with the actual path where your X2C dataset is stored.

⚙️ **Configuration Reminder**

Update the **ictrl_data_path** field in your config.yaml to point to your local copy of the X2C dataset.

## Mapping Network Training
```train
python main.py train.batch_size=128 train.num_workers=16 train.num_epochs=100 train.lr=1e-3
```

## Mapping Network Evaluation
```eval
python main.py do_eval=True train.batch_size=128 train.num_workers=16 train.save_model_path=path/to/save_folder
```

## 📥 Pre-trained Models
You can download pre-trained models here:

 [🔗Mapping Network](https://drive.google.com/file/d/1GAiBihDk-vcc-wK-GY5o-kwWobUA4g53/view?usp=sharing) trained on <strong>X2C</strong> with a batch size of 128, learning rate of 1e-3, for 100 epochs, using ResNet18 as the feature extractor.

## 🚀 X2CNet Inference Pipeline

Download the required checkpoints for the **motion transfer module** from [LivePortrait](https://github.com/KwaiVGI/LivePortrait).

Update the paths in [`liveportrait_configs/inference_config.py`](liveportrait_configs/inference_config.py) accordingly.

To generate control values for on-robot execution, run:

```bash
python x2cnet_inference.py --driving /path/to/driving_video
```


## Real-world Inference Results
![Alt text](docs/static/images/inference_example2.png)


## 🤝 Contributing
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.


## 💖 
If you find <strong>X2C</strong> or <strong>X2CNet</strong> useful for your research, welcome to 🌟 this repo 




