## Dataloader 사용법

~~~python
trn_loader, dev_loader, tst_loader = get_dataloader(
    csv_path=args.csv_path,
    batch_size=args.batch_size,
    label_type=args.label_type,
    iid=args.iid,
    transform=args.norm_type,
)
~~~

- 아래 파라미터들은 모두 config에 있는 argparse로 변경 가능
- csv_path (str): image path와 label 정보를 담고 있는 csv 파일 위치
- batch_size (int): 본인의 PC 환경에 맞게끔 구성
- label_type (int): 파고(0), 파향(1), 파주기(2) 중 선택
- iid (int): 0, 1, 2, 3, 4 총 5개의 label 존재 -> 아래 내용을 반드시 확인 하세요!
- transform (int): normalization 방식 선택(option: [0, 1, 2])
    - 0: normalization 없이 ToTensor()만 진행
    - 1: 전체 이미지의 평균/표준편차로 normalization 진행(mean: 0.3352, std: 0.0647)
    - 2: image by image normalization 진행

### iid 지정 방법
- 기존
    - parser에서 boolean으로 지정
- 변경
~~~python
# 기존 iid 부분을 아래와같이 변경
# 이 부분은 각자 사용하는 아그파서가 달라서 제가 일괄적으로 처리하지 못하였습니다.
parser.add_argument(
    "--iid",
    default=None,
    help="use argument for iid condition",
)
~~~

- 사용법
~~~python
# 아래와같이 --iid 뒤에 사용할 레이블 인덱스를 추가할 것
# choices = [0, 1, 2, 3, 4]
python cnn_regressor/oned.py --norm-type 2 --batch-size 512 --root-img-path /media/lepoeme20/Data/projects/daewoo/ --dataset weather_4 --label-type 0 --iid 0
~~~