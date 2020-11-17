## Regression 진행법

**모든 코드는 최상위 폴더(main.py 가 있는 경로)에서 실행됩니다**

### 1) Auto-encoder 진행
```bash
python ae_regressor/00_train_ae.py --norm-type 2 --batch-size 256 --iid --data-type 0
```

지정 가능한 파라미터(기본값의 경우 미지정시 자동으로 사용되는 값을 의미함):
- csv-file: data path와 label이 들어 있는 csv file의 경로(기본값: './preprocessing/brave_data_label.csv')
- img-size: auto encoder의 input image size(기본값: 32)
- norm-type: Normalization 방식, **반드시 지정 필요**
    - 0: ToTensor
    - 1: Ordinary image normalizaeion
    - 2: Image by Image normalization
- batch-size: Mini batch size 지정, **반드시 지정 필요**
- epochs: 학습 에폭 지정(기본값: 30)
- log-interval: log를 print하는 주기 지정(기본값: 200)
- iid: data를 i.i.d. condition으로 생성하고 싶은 경우 기재 **기재하지 않을 경우 time series dataset으로 구성**
- test: 학습이 완료된 모델을 test하고 싶은 경우 기재 **기재하지 않을 경우 training 진행**
- data-type: 파향, 파고, 파주기 선택 **반드시 지정 필요**
    - 0: 파고
    - 1: 파향
    - 2: 파주기
- cae: Auto encoder 형태 지정 **기재하지 않을 경우 linear auto encoder로 학습 진행**

```bash
# 다른 argument의 경우 위의 예제와 같이 --argument-name value 로 지정 가능
# iid와 test의 경우 기재 자체가 값을 가짐

# 일반적인 normalization + 256 batch size + iid dataset으로 학습을 진행하고 싶은 경우
python ae_regressor/00_train_ae.py --norm-type 1 --batch-size 256 --iid

# 일반적인 normalization + 256 batch size 옵션으로 학습된 모델을 테스트 하고싶은 경우
python ae_regressor/00_train_ae.py --norm-type 1 --batch-size 256 --test
```

### 2) Regression 진행
```bash
python ae_regressor/01_regression.py --norm-type 1 --batch-size 1024 --num-parallel -1 --sampling-ratio 0.1
```
- batch-size: 모델을 학습하는 것이 아니라 latent code를 가져오기 위함으로 큰 배치를 사용해도 메모리 문제가 없음
- num-parallel: sklearn grid search를 위한 process 수, -1을 지정할 경우 사용 가능한 모든 프로세스를 사용하므로 적당한 수 지정필요. 본인의 cpu core 수를 알면 -1 혹은 -2정도 하면 적당. 예를들어 octa-core의 경우 7 혹은 6으로 지정
- sampling-ratio: 전체 데이터 중 몇 퍼센트로 샘플링하여 grid search를 진행할지 정하는 파라미터
나머지 지정 가능한 파라미터와 사용 방법은 상기 1) 과 동일함
