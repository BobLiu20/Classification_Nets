### Implement popular models by tensorflow

 1. Using two different API:
 * **train_estimator.py**: Using tf.estimator API
 * **train_low_level.py**: Using low level API

 2. Test in tensorflow 1.3.0

### How to training

 1. Prepare dataset
 ```
 cd datasets
 python download_and_convert_data --dataset_name=cifar10 --dataset_dir=/your/path
 ```

 2. Training
 Please using help to get args list
 ```
 cd training
 python train_estimator.py --help 
 ```

 3. Testing
 Please using help to get args list
 ```
 cd testing
 python eval_image_classifier.py --help 
 ```

### Reference
 These code reference to tensorflow [slim](https://github.com/tensorflow/models/tree/master/research/slim)

