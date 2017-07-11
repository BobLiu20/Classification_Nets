### What
Testing all nets in cifar10 in order to compare them.

### How to use

##### 1. Prepare cifar data
  Open command line and enter data folder.    
  ```
  export CAFFE=/your/caffe/path/    
  ./get_create_cifar10.sh    
  ```
  Please set CAFFE env to caffe's folder and then run script.    

##### 2. Training
  Enter one of nets. And then run:    
  ```
  caffe train --solver=solver.prototxt --gpu=0    
  ```
  eg:    
  ```
  cd ResNet20    
  caffe train --solver=solver.prototxt --gpu=0    
  ```

### Nets
  Kind in mind, it is not any augmentation in training.    
  The image size of cifar10 is 32x32.    

  caffe: 1.0.0-rc3    
  cuda: 8.0    
  nvidia: GTX1080 ti (Total memory is 11GB)    
  system: ubuntu 14.04 in docker    

* AlexNet    
  The size of model: 351KB    
  The accuracy of testing is 0.7823 after 64000 Iterations.    
  The time of training is 0.5 hours in GTX1080.    

* ResNet20    
  The size of model: 1.1MB    
  The accuracy of testing is 0.8258 after 64000 Iterations.    

* ResNet32    
  The size of model: 1.8MB    
  The accuracy of testing is 0.8794 after 64000 Iterations.    

* ResNet56    
  The size of model: 3.4MB.    
  The accuracy of testing is 0.8706 after 64000 Iterations.    
  The memory usage of GPU is 3.8GB in training. (batch size is 128).    
  The time of training is 15 hours in GTX1080.    

* WRN28_10: [Wide Residual Networks](http://arxiv.org/abs/1605.07146)    
  The size of model: 140MB.    
  The accuracy of testing is 0.8950 after 60000 Iterations.    
  The memory usage of GPU is 9.9GB in training. (batch size is 128).    
  The time of training is 22.5 hours in GTX1080 ti.    

* VGG16    
  The size of model: 129MB.    
  The accuracy of testing is 0.8308 after 64000 Iterations.    
  The memory usage of GPU is 1GB in training. (batch size is 128).    
  The time of training is 1 hours in GTX1080 ti.    

* GoogLeNet    
  The size of model: 25MB.    
  The accuracy of testing is 0.7913 after 64000 Iterations.    
  The memory usage of GPU is 1.3GB in training. (batch size is 128).    
  The time of training is 1 hours in GTX1080 ti.    

