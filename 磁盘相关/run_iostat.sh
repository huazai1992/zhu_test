echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_2.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_3.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_4.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_5.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_6.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_7.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_8.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
iostat -x 1 3600 >> ./results/iostat2240_11000_9.txt & (python ./test_io.py --batch-size 11000 --num-epoch 1 --num-size 2240&& ps aux| grep "iostat" | grep -v grep | cut -c 9-15 | xargs kill -9)
echo 1 > /proc/sys/vm/drop_caches
sleep 1
