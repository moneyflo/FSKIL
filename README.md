# Few-shot Keyword-incremental Learning with Total Calibration

## Prerequisites

- Download package
```
python3 -m venv venv
. ./venv/bin/activate
pip install torch torchvision torchaudio tqdm matplotlib scikit-learn requests
```

- Data processing
```
cd data
python3 dataset_prep.py
cd ../
```

- Start training
```
. ./scripts/gsc2.sh
```

## Dataset
Using Google Speech Commands v2


## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [bcresnet](https://github.com/Qualcomm-AI-research/bcresnet)
- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)
- [TEEN](https://github.com/wangkiw/TEEN)