# Vision Model Practice Home Repository
- https://github.com/hyunwook990/vision_model_practice.git
# VGG 정리(Notion)
- https://skitter-airport-cf1.notion.site/ResNet-1b708a64f175810f8fa0c97de4969b16
---
# VGG_practice
## 2025-04-18
- VGG 논문을 읽고 모델의 구조(layer만)를 구현해보기
- 1차 구현 완료

## 2025-04-28
- fully-connected 구조를 제대로 알고 다시 수정함.
- ImageNet 데이터셋을 사용하지 않고 임의의 데이터셋을 사용하여 모델을 학습하고 검증.

## 2025-04-30
- kaggle의 butterfly image를 데이터셋으로 결정, vgg16을 구현해 분류를 진행
    - (출처: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
- 데이터로드, 역전파 구현 목표

### 문제점
1. CIFAR-10을 불러올때에는 라이브러리로 불러왔는데 이번에 kaggle에서 다운받은 이미지를 넣으려니까 입력 형식이 달라서 어떻게 해야할지 몰라서 문제가 발생
2. 1번의 연장선으로 train의 dataloader가 없어서 역전파 구현이 완벽하지 않음
3. 논문에 loss함수는 무엇을 사용했는지 없어서 구현을 못함.

## 2025-05-04
- 데이터 저장, 데이터 로드 코드를 분석 및 공부, 모델 학습까지 목표
- 1차 중단점 19:26 데이터 저장 완료, 데이터 로드 구현중
- 학습함수 이전까지 완료, Early Stopping, BestScoreSave 분석 필요