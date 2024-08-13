# HMG_Hanyang_Lane_Route_Validation - Transformer
Transformer based Codes for HMG-Hanyang Lane-level Route Validation Project

## Highlight
- Transformer 기반 차로 유효성 판단 Repository
    - 크게 3 파트로 나누어 학습 및 평가 진행
        - Pre-training part, Fine-tuning part, Evaluation part
- 트랜스포머 기반 베이스 라인 모델 : <a href="https://github.com/jchengai/forecast-mae">Forecast-MAE</a>
- 검증을 위한 dataset과 pre-trained 모델도 함께 업로드

## Dataset Generation and Path Setting
1. Matlab 레포지토리 RunTRANSFORMERBasedDatasetGen.m 실행시켜 HMG 양식의 데이터셋 생성
2. Train_dataset, Validation_dataset, Test_dataset을 하나의 폴더에 넣기
    - 학습 및 추론 시, 폴더 위치 접근 후, 폴더 내 Train_dataset.mat, Validation_dataset.mat, Test_dataset.mat 접근하는 구조

## Training
### Phase 1 - pre-training:
#### SSL(Self-supervised Pre-training)을 통한 Embedding 값 사전 학습
```
python3 train.py data_root=/path/to/data_root model=model_mae batch_size=128
```
- 실제 데이터셋 위치를 /path/to/data_root에 기입
    - ex) python3 train.py data_root=/Desktop/git/HMG_Transformer model=model_mae batch_size=128

- 학습 완료 시, ```Outputs/forecast-mae-pretrain/YYYY-MM-DD/HH-MM-SS``` 폴더 내 학습 결과 생성
    - ex) Outputs/forecast-mae-pretrain/2024-08-13/11-40-41 폴더 생성 확인

### Phase 2 - fine-tuning:
#### Phase 1을 통해 pre-trained된 모델을 통한 학습 진행
```
python3 train.py data_root=/path/to/data_root model=model_forecast batch_size=128 'pretrained_weights="/path/to/pretrain_ckpt"'
```
- 실제 데이터셋 위치를 /path/to/data_root, Pre-training 모델 결과를 /path/to/pretrain_ckpt에 기입
    - ex) python3 train.py data_root=/Desktop/git/HMG_Transformer model=model_forecast batch_size=128 'pretrained_weights="/Desktop/git/HMG_Transformer/Outputs/2024-08-13/11-40-41/checkpoints/last.ckpt"'

- 학습 완료 시, ```Outputs/forecast-mae-forecast/YYYY-MM-DD/HH-MM-SS``` 폴더 내 학습 결과 생성
    - ex) Outputs/forecast-mae-forecast/2024-08-13/12-49-19 폴더 생성 확인

## Evaluation
#### 학습된 모델을 통해 실제 추론하고자하는 Test_dataset Route Validation 진행
```
python3 eval.py data_root=/path/to/data_root model=model_forecast batch_size=64 'checkpoint="/path/to/checkpoint"'
```
- 실제 데이터셋 위치를 /path/to/data_root, Phase 2 모델 결과를 /path/to/pretrain_ckpt에 기입
    - ex) python3 eval.py data_root=/Desktop/git/HMG_Transformer model=model_forecast batch_size=128 'checkpoint="/Desktop/git/HMG_Transformer/Outputs/2024-08-13/12-49-19/checkpoints/last.ckpt"'

- 학습 완료 시, command 창에 나오는 ```Accuracy, Precision, Recall``` 확인


## Acknowledgements

This repo benefits from [MAE](https://github.com/facebookresearch/mae), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [NATTEN](https://github.com/SHI-Labs/NATTEN) and [HiVT](https://github.com/ZikangZhou/HiVT). Thanks for their great works.
