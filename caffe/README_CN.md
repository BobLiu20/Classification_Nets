[Switch to English ReadMe](https://github.com/BobLiu20/Classification_Nets/blob/master/README.md)    

### 这是什么    
在Cifar10数据集上，测试所有分类网络，对比他们的各种性能    
可以在这里找到基本上所有经典的分类网络    

### 如果使用    

##### 1. 准备Cifar10训练数据    
  打开命令行，cd到data文件夹,然后执行    
  ```
  export CAFFE=/your/caffe/path/    
  ./get_create_cifar10.sh    
  ```
  请设置环境变量CAFFE到caffe源码路径(已编译好的), 再运行脚本。    

##### 2. 开始训练    
  进入其中任意一个网络，然后执行下面命令开始训练    
  ```
  caffe train --solver=solver.prototxt --gpu=0    
  ```
  例如:    
  ```
  cd ResNet20    
  caffe train --solver=solver.prototxt --gpu=0    
  ```

### 各种分类网络    
  需要注意的是，下面的训练都没有进行数据扩充操作。(如果有做，会进一步提高精度)    
  图片的输入尺寸是32x32.    

  caffe: 1.0.0-rc3    
  cuda: 8.0    
  nvidia: GTX1080 ti (Total memory is 11GB)    
  system: ubuntu 14.04 in docker    

* LeNet    
  模型大小: 351KB    
  测试精度: 0.7823 after 64000 Iterations.    
  训练耗时: 0.5 hours in GTX1080.    
  预测耗时: 0.93 ms. (one image with 32x32)    

* BN-LeNet: Batch Normalization LeNet    
  模型大小: 352KB    
  测试精度: 0.7935 after 64000 Iterations.    
  预测耗时: 0.76 ms. (one image with 32x32)    

* AlexNet    
  模型大小: 3MB    
  测试精度: 0.7452 after 64000 Iterations.    
  训练耗时: 10 mins in GTX1080.    
  预测耗时: 0.65 ms. (one image with 32x32)    

* SqeezeNet_v1.1    
  模型大小: 2.8M    
  测试精度: 0.8114 after 64000 Iterations.    
  预测耗时: 2.2 ms. (one image with 32x32)   

* NetworkInNetwork: NIN    
  模型大小: 25MB   
  测试精度: 0.8346 after 64000 Iterations.    
  训练耗时: 40 mins in GTX1080.    
  预测耗时: 1.8 ms. (one image with 32x32)    

* ResNet20    
  模型大小: 1.1MB    
  测试精度: 0.8258 after 64000 Iterations.    
  预测耗时: 4.2 ms. (one image with 32x32)    

* ResNet32    
  模型大小: 1.8MB    
  测试精度: 0.8794 after 64000 Iterations.    
  预测耗时: 7.8 ms. (one image with 32x32)    

* ResNet56    
  模型大小: 3.4MB.    
  测试精度: 0.8706 after 64000 Iterations.    
  训练时显存消耗: 3.8GB in training. (batch size is 128).    
  训练耗时: 15 hours in GTX1080.    
  预测耗时: 16.3 ms. (one image with 32x32)   

* WRN28_10: [Wide Residual Networks](http://arxiv.org/abs/1605.07146)    
  模型大小: 140MB.    
  测试精度: 0.8950 after 60000 Iterations.    
  训练时显存消耗: 9.9GB in training. (batch size is 128).    
  训练耗时: 22.5 hours in GTX1080 ti.    
  预测耗时: 13.0 ms. (one image with 32x32)    

* VGG16    
  模型大小: 129MB.    
  测试精度: 0.8308 after 64000 Iterations.    
  训练时显存消耗: 1GB in training. (batch size is 128).    
  训练耗时: 1 hours in GTX1080 ti.    
  预测耗时: 4.6 ms. (one image with 32x32)    

* GoogLeNet    
  模型大小: 25MB.    
  测试精度: 0.7913 after 64000 Iterations.    
  训练时显存消耗: 1.3GB in training. (batch size is 128).    
  训练耗时: 1 hours in GTX1080 ti.    
  预测耗时: 7.2 ms. (one image with 32x32)    

* DenseNet    
  模型大小: 4MB.    
  测试精度: 0.9153 after 64000 Iterations.    
  训练时显存消耗: 7.9GB in training. (batch size is 32+32).    
  训练耗时: 3.5 hours in GTX1080 ti.    
  预测耗时: 12.95 ms. (one image with 32x32)    



