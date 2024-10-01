# Expression or Emotion - Preserving Face Dataset Anonymization via Latent Code Optimization.

This project aims to provide a model that can transform a dataset of human faces into an anonymized version of that datasset while maintaining the same facial expression.

## Example

[d081ea83-0a5f-4ba3-be71-30152e6d9726](https://github.com/user-attachments/assets/23e3e818-3d91-41b8-9ca1-918834fddfbe)

## Installation
This project was developed on linux, there is no guarantee it will run as intended on other operating systems.
Important information to note:

* This project was developed using CUDA toolkit version 11.8, while developing it would refuse to work with version 11.5.
* Python version: 3.7.17
* Pyenv and Poetry were used to manage the python version, a file .python-version file is included. 

### Create the Environment:
```
# install poetry if not already on the system.
pip install poetry

# Create and enter the virtual environment
poetry shell

# install dependencies from the pyproject.toml file
poetry install
```

### Dataset:
* The CelebA-HQ dataset can be found at: https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view
* Also download the train/test/val official txt file from: https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view
* Create a new directory in the root directory called "datasets/real".
* In the "real" folder, create a folder called "annotations" and extract the "CelebA-HQ-to-CelebA-mapping.txt" there.
* Place the "list_eval_partition.txt" in the same "annotations" folder
* Place the "CelebA-HQ-img", which contains the 30,000 images, into the "real" directory 

The Directory should look like this:
```
datasets/real
├── annotations
│   ├── CelebA-HQ-to-CelebA-mapping.txt
│   └── list_eval_partition.txt
└── CelebA-HQ-img
    ├── 0.jpg
    ├── 1.jpg
    ├── ...
    ├── 9998.jpg
    └── 9999.jpg
```

### Download the required pretrained models
```
python download_models.py
```

## Use the model in the following order to get expected results
### Resize images for landmarks extractor
```
python resize_images.py
```
### Extract landmarks, identity features and latent code
```
python extract_features.py
```
### Generate fake dataset
```
python generate_fake_data.py
```
### Use Nearest Neighbour model to find closest real and fake images
```
python pair_nn.py
```
### Save inverted latent code of real images
```
python invert.py
```
### Create the final anonymized images
```
python anonymize.py
```
### Other
* In case you wish to change the directories or experiment with differing parameters, make modifications to the config.yaml file.
* Call methods definded within "lib/evaluation.py" using the "evaluation.ipynb" notebook toe valuate the outputs.

## Disclaimer
This project was built as part of my dissertation submission for my MSc in Artificial Intelligence. This work is modification and based on the work researched and developed at https://github.com/chi0tzp/FALCO .
