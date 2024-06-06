# Makeup-Openl

Mindspore implementation of the TMM 2023 paper [DRAN: Detailed Region-Adaptive Normalization for Conditional Image Synthesis](https://arxiv.org/abs/2109.14525) .

## Dependencies

* Python 3.7
* mindspore 2.1.1
* mindcv 0.2.2
* opencv-python 4.9.0.80，opencv-python-headless  4.9.0.80
* requests 2.25.1
* fvcore 0.1.5.post20221221
* fire 0.5.0
* face_recognition

## Test

Checkpoints have been provided in [checkpoints/32_3073_G.ckpt](https://openi.pcl.ac.cn/hnu202004060702/Makeup-Openl2/src/branch/master/checkpoints) .

For test, you should first put your source images and reference images into **`datasets/source_test`** and **`datasets/reference_test`**  respectively, and then run：

```shell
python demo_test_all.py
```

## Results

One example：

![img](https://openi.pcl.ac.cn/hnu202004060702/Makeup-Openl2/raw/branch/master/results/full_general/vSYYZ836_vFG15.png)

In this code, we utilized 68 facial landmarks. Using more landmarks can get better results.