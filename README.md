# YOLO26 Pruning

YOLO26(Ultralytics 기반) 모델에 pruning을 적용하고, 압축된 모델을 재학습/평가하기 위한 실험용 프로젝트입니다.

## 주요 내용

- `yolo26/compress/Compress.py`: pruning 적용 및 모델 구조 재구성 로직
- `yolo26/compress/GM.py`: GM(Geometric Median) 기반 structured pruning 구현
- `yolo26/prune_finetune.py`: pruning 후 fine-tuning 실행 스크립트
- `yolo26/train_baseline_model.py`: baseline 모델 학습 스크립트
- `yolo26/eval.py`: 모델 평가 스크립트

## 설치

```bash
cd yolo26
pip install -r requirements.txt
pip install -e .
```

## 데이터셋

데이터셋은 루트의 `dataset/` 디렉터리에 배치하고, 경로 및 클래스 정보는 `dataset.yaml`에서 관리합니다.

`dataset/`은 용량이 큰 파일이므로 git에 포함하지 않습니다.

## Pruning + Fine-tuning 실행 예시

```bash
python yolo26/prune_finetune.py \
  --bmodel yolo26n.pt \
  --compress_ratio 0.5 \
  --prune_type H \
  --method L2 \
  --cfg_output_path pruned_yolo26.yaml \
  --epoch 50 \
  --name yolo26_pruned
```

주요 옵션:

- `--compress_ratio`: pruning 비율
- `--prune_type`: pruning 대상 (`H`, `B`, `ALL`)
- `--method`: pruning 방법 (`L1`, `L2`, `GM`)
- `--bmodel`: baseline 또는 pretrained 모델 가중치 경로
- `--resume_path`: 중단된 학습 재개용 checkpoint 경로

## Git 관리 제외 항목

데이터셋, checkpoint, 모델 가중치(`*.pt`, `*.pth` 등), export 결과(`*.onnx`, `*.engine` 등), 캐시 파일은 `.gitignore`에 등록되어 있습니다.
