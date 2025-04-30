# Vision Model Practice Home Repository
- https://github.com/hyunwook990/vision_model_practice.git
# VGG 정리(Notion)
https://skitter-airport-cf1.notion.site/ResNet-1b708a64f175810f8fa0c97de4969b16
---
# VGG_practice
### 2025-04-18
- VGG 논문을 읽고 모델의 구조(layer만)를 구현해보기
- 1차 구현 완료

### 2025-04-28
- fully-connected 구조를 제대로 알고 다시 수정함.
- ImageNet 데이터셋을 사용하지 않고 임의의 데이터셋을 사용하여 모델을 학습하고 검증.

### 2025-04-30
- kaggle의 butterfly image를 데이터셋으로 결정, vgg16을 구현해 분류를 진행
    - (출처: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
- 데이터로드, 역전파 구현 목표