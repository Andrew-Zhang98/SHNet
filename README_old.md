# SHNet

## Prerequisites
- [Python 3.7.11](https://www.python.org/)
- [Pytorch 1.7.1](http://pytorch.org/)
- [OpenCV 4.5.3](https://opencv.org/)
- [Numpy 1.21.2](https://numpy.org/)
- [pillow 8.2.0](https://pypi.org/project/Pillow/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [fvcore](https://github.com/facebookresearch/fvcore)

## Download dataset
Download the following datasets and unzip them into `/Data/RGB-SOD/` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

Your `/Data/RGB-SOD/` folder should look like this:
````
-- RGB-SOD
   |-- DUTS
   |   |-- DUTS-TE
   |   |-- | images
   |   |-- | GT
   |-- ECSSD
   |   |--images
   |   |--GT
   ...
````

## Testing & Evaluate
- Pre-computed saliency maps for DUTS-TE, ECSSD, DUT-OMRON, PASCAL-S, and HKU-IS datasets and trained resnet-based model are here: [Google drive](https://drive.google.com/drive/folders/1hZ8EGelTVDFDWjC0bwxxzmgsdRp6LfL7) | [Baidu YunPan(194y)](https://pan.baidu.com/s/1zsaZiId-3n9665uAZmlcow).
- If you want to evaluate the performance of our work, please download datasets, our trained model and our code, then put the datasets and model to the corresponding path. 
- Predict the saliency maps
```shell
    sh launsh.sh
```
- If you want to evaluate the predicted results, please check [PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit) and follow the instruction.
