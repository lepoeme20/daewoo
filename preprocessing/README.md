## DSME Preprocessing Code

### Data transfer
~~~bash
# passwd: 1234
# weather_4 data
rsync -avz lepoeme20@163.152.182.224:/media/lepoeme20/Data/projects/daewoo/use_data/weather_4 your/local/data/path

# weather_1 data
rsync -avz lepoeme20@163.152.182.224:/media/lepoeme20/Data/projects/daewoo/use_data/weather_1 your/local/data/path

# original data
rsync -avz lepoeme20@163.152.182.224:/media/lepoeme20/Data/projects/daewoo/weather/data your/local/data/path

# label files
label files are located in preprocessing folder
~~~

- rsync: 데이터 전송 명령어
    - rsync -avz 대신 scp -r 사용 가능
- your/local/data/path에는 데이터를 저장할 local path를 입력할 것
    Desktop/ 으로 지정 시, Desktop/data_crop 이 생성됨
- 비밀번호는 1234 입력 후 엔터

