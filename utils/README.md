## Dataloader 사용법

~~~python
trn_loader, dev_loader, tst_loader = get_dataloader(
    csv_path=,
    batch_size=,
    iid=,
    transform=
)
~~~

- csv_path (str): image path와 label 정보를 담고 있는 csv 파일 위치
- batch_size (int): 본인의 PC 환경에 맞게끔 구성
- iid (bool): True or False
- transform (int): normalization 방식 선택(option: [0, 1, 2])
    - 0: normalization 없이 ToTensor()만 진행
    - 1: 전체 이미지의 평균/표준편차로 normalization 진행(mean: 0.3352, std: 0.0647)
    - 2: image by image normalization 진행