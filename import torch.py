import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T   #이미지 전처리 및 증강 도구

#Launch TensorBoard Session
from torch.utils.tensorboard import SummaryWriter   #TensorBoard에 손실값, 정확도, 이미지등을 기록하는 도구
from torch.utils.data import DataLoader   #데이터셋을 배치 단위로 나눠서 모델에 공급 역할
from torchvision.datasets import CIFAR100   #100개 클래스, 총 60,000장의 이미지로 구성된 벤치마크 데이터셋
from torchmetrics.aggregation import MeanMetric   #여러 배치에 걸친 값의 평균을 누적 계산
from torchmetrics.functional.classification import accuracy   #예측값과 정답을 비교해 정확도를 계산하는 함수


#Build config: 소프트웨어를 빌드 할때 필요한 설정들을 모아놓은 것
title = 'resnet101_finetune'
device = 'cuda'   #GPU 사용할때
root = 'data'   #학습 데이터가 저장된 폴더 경로
batch_size = 128  # 한번에 모델에 넣는 이미지 수
num_workers = 8
lr = 0.01
weight_decay = 1e-8    #과적합 방지용 정규화 강도->가중치가 너무 커지는 것을 억제
label_smoothing = 0.05 #정답 레이블을 살짝 부드럽게 만들어 과적합 방지
epochs = 20
log_dir = 'logs'   #TensorBoard 로그 저장 폴더
checkpoint_dir = 'checkpoints'   #학습 중간중간 모델 가중치를 저장하는 폴더
pretrained_model = 'resnet101-cd907fc2.pth'  #사전 학습된 ResNet101가중치 파일-> 나중에 이것을 불러와서 fine tuning



##########
#Build dataset: 이미지 전처리   이미지를 변환하여 원본과 다르게 바꿈
train_transform = T.Compose([
    T.RandomCrop(size=(32, 32), padding=4),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),
    T.ToTensor(), #0~255 -> 0.0~1.0
    T.RandomErasing(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#train데이터와 validation데이터셋 만들기
train_data = CIFAR100(root, train=True, download=True, transform=train_transform)
trian_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)  #shuffle=True는 매 에폭마다 데이터 순서를 섞음(과적합 방지), drop_last=True는 마지막 배치가 batch_size보다 작으면 버림

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_data = CIFAR100(root, train=False, downlad=True, transform=val_transform)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)