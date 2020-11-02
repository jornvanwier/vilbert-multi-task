# Installation
Make sure cuda and up to date nvidia drivers are installed.

Clone the repository and create the environment:
```
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
# git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
git clone --recursive https://github.com/jornvanwier/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt
```

Install pytorch, cuda with conda:
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

_OPTIONAL_ Install apex (in another directory, but same conda environment):
```
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Feature extractor:
```
cd ..
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
pip install ninja yacs cython matplotlib

# Download models
cd ../vilbert-multi-task/data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```


# Data

## Flickr30k TASK 18 dataset
__not needed for interactive grounding script__

Download (90+ GB): 
```
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz
```

Extract needed data from this.

Update paths until it finds what its looking for.