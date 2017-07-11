#!/usr/bin/env sh
set -e

# 0. Request Caffe's folder

if [ -z $CAFFE ]; then
echo "Please set env CAFFE to your caffe path."
echo "eg: export CAFFE=/your/caffe/path"
exit 0
fi

# 1. Download data

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."

wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

tar -xf cifar-10-binary.tar.gz && rm -f cifar-10-binary.tar.gz

# 2. Convert data

EXAMPLE=./
DATA=./cifar-10-batches-bin/
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

$CAFFE/build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

$CAFFE/build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

rm -rf cifar-10-batches-bin

echo "Done."
