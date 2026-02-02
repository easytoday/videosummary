## Prepare your working environment
### 1. install requirements with conda
You need some dependencies, please install them as Follow this template.
```bash
conda env create --name envname --file=requirements.yml
```
### 2. Download preprocessed datasets
[google drive link][https://drive.google.com/open?id=1Bf0beMN_ieiM3JpprghaoOwQe9QJIyAN]

### 3.Download the alternative preprocessed dataset 
[alt_dataset][https://www.kaggle.com/datasets/c572e52ed35f22090e1d4b8a304c8b81c5411cef539030663b2b75bfd193f2e7]


### 4. Make splits
```bash
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir datasets --save-name summe_splits  --num-splits 5
```
As a result, the dataset is randomly split for 5 times, which are saved as json file.

Train and test codes are written in `main.py`. To see the detailed arguments, please do `python main.py -h`.

## How to train
```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --verbose
```

## How to test
```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --evaluate --resume path_to_your_model.pth.tar --verbose --save-results
```

If argument `--save-results` is enabled, output results will be saved to `results.h5` under the same folder specified by `--save-dir`. To visualize the score-vs-gtscore, simple do
```bash
python visualize_results.py -p path_to/result.h5
```


## Citation
```
@article{zhou2017reinforcevsumm, 
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao}, 
   journal={arXiv:1801.00054}, 
   year={2017} 
}
```
