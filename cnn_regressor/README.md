Use oned.py
~~~python
# Change img path in csv file
# Run only at the first time
python cnn_regressor/change_img_path.py --root-img-path /media/lepoeme20/Data/projects/daewoo/ --dataset weather_4

# Training
python cnn_regressor/oned.py --norm-type 2 --batch-size 512 --label-type 0 --root-img-path /media/lepoeme20/Data/projects/daewoo/ --trn-dataset weather_4 --iid 0 --stride 1

# Inference
python cnn_regressor/oned.py --norm-type 2 --batch-size 512 --label-type 0 --root-img-path /media/lepoeme20/Data/projects/daewoo/ --trn-dataset weather_4 --iid 0 --stride 1 --tst-dataset weather_1 --test
~~~