python tf_cnn_benchmarks.py --data_dir=./test_data/imagenet_224 --num_gpus=1 --batch_size=10 --model=alexnet --variable_update=independent --forward_only=True --local_parameter_device=gpu
echo next >> monitor.out
