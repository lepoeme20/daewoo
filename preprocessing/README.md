## DSME Preprocessing Code

### Data transfer
~~~bash
# passwd: 1234
# weather data
rsync -avz lepoeme20@163.152.182.224:/media/lepoeme20/Data/projects/daewoo/weather/data_crop your/local/data/path
# weather original label file
rsync -avz lepoeme20@163.152.182.224:/media/lepoeme20/Data/projects/daewoo/weather/wavex_11.csv your/local/data/path
~~~

- rsync: 데이터 전송 명령어
    - rsync -avz 대신 scp -r 사용 가능
- your/local/data/path에는 데이터를 저장할 local path를 입력할 것
    Desktop/ 으로 지정 시, Desktop/data_crop 이 생성됨
- 비밀번호는 1234 입력 후 엔터

### Create label file
~~~bash
cd preprocessing
# code는 preprocessing folder에서 실행할 것
python get_label.py --dataset weather --data-path /your/local/data/path/data_crop --radar-path /your/local/data/path/wavex_11.csv
~~~

- label file 생성
    - 이전 logic과 동일하게 앞뒤 2:30/ 총 5분간 같은 label 부여
    - column명 이전과 동일(iid, time 별 phase 존재)
        - **Trn, dev, tst는 반드시 파일 내부에 있는 phase column을 사용하여 split할 것**
    - label file은 preprocessing folder 내부에 생성됨
