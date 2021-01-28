Use oned.py
~~~python
# Training
python cnn_regressor/oned.py --norm-type 2 --batch-size 512 --label-type 0 --root-img-path /media/lepoeme20/Data/projects/daewoo/ --trn-dataset weather_4 --iid 0 --stride 1

# Inference
python cnn_regressor/oned.py --norm-type 2 --batch-size 512 --label-type 0 --root-img-path /media/lepoeme20/Data/projects/daewoo/ --trn-dataset weather_4 --iid 0 --stride 1 --tst-dataset weather_1 --test
~~~