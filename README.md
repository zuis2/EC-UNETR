# EC-UNETR

Source code of  "Ensemble Cross UNet Transformers for Augmentation of Atomic Electron Tomography"
[Project] https://github.com/zuis2/EC-UNETR

- We used the monai package, you can find it in "thirdparty" folder, or install it by "pip install monai"

## Dataset

### Download datasets from Baidu disk:
    dataset link:  https://pan.baidu.com/s/1yjDck2m8ss31ifwhIVSyUw 
    key: kikx 


### Unzip to a folder:
```shell
# root path of the datasets 
datasets/
├── BF5_FCC 
├── BF3.2_FCC
└── BF5_Amor
```

## Config
If you want to change any training or testing parameter,  please check the  config file in our source code folder:

```shell
config/
└── config.json
```

## Train

### Set config
```json
{
    "model_name":"EC_UNETR_W",      //model name
    "datasets_path":"path_of_dataset", //root path of the datasets
    "dataset_name":"BF5_FCC",      //dataset name BF5_FCC, BF3.2_FCC, BF5_Amor,
    "batch_size":1,
    "gpu_id":"",
    "start_epoch":0,  //only change for resume
    "N_epoch":100,    // epoch
    "resume_state":"",//path of resume state
    "output_file":"",//for output 
    "output":false,//for output
    "train":true,
    "test":false,
    "save_folder":"./experiments/",
    "block_size":144
}
```

### Run:
```python
    python main.py
```

## Test

### Set config
```json
{
    "model_name":"EC_UNETR_W",      //model name
    "datasets_path":"path_of_dataset", //root path of the datasets
    "dataset_name":"BF5_FCC",      //dataset name BF5_FCC, BF3.2_FCC, BF5_Amor,
    "batch_size":1,
    "gpu_id":"",
    "start_epoch":0,  //only change for resume
    "N_epoch":100,    // epoch
    "resume_state":"path_of_pretrained_model",//path of resume state
    "output_file":"",//for output 
    "output":false,//if you want to  output augmented tomograms in test mode, use 'true' here
    "train":false,
    "test":true,
    "save_folder":"./experiments/",
    "block_size":144
}
```

### Run:
```python
    python main.py
```

### Download pre-trained model from Baidu disk:
    dataset link:  https://pan.baidu.com/s/1Hglt_UgZWpKwesAyHzvd6A 
    key: qme9

## Augment a single raw tomogram

### Set config
```json
{
    "model_name":"EC_UNETR_W",      //model name
    "datasets_path":"path_of_dataset", //root path of the datasets
    "dataset_name":"BF5_FCC",      //dataset name BF5_FCC, BF3.2_FCC, BF5_Amor,
    "batch_size":1,
    "gpu_id":"",
    "start_epoch":0,  //only change for resume
    "N_epoch":100,    // epoch
    "resume_state":"path_of_pretrained_model",//path of resume state
    "output_file":"path_of_input_tomogram",//for output 
    "output":false,//
    "train":false,
    "test":false,
    "save_folder":"./experiments/",
    "block_size":144
}
```

### Run:
```python
    python main.py
```
