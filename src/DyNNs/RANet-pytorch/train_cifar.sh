python main.py \
  --arch RANet --gpu '0' \
  --data-root '/home/sxc180080/data/Dataset/CIFAR10' \
  --data 'cifar10' \
  --step 4 \
  --stepmode 'even' \
  --scale-list '1-2-3-3' \
  --grFactor '4-2-1-1' \
  --bnFactor '4-2-1-1' \
  --save  'save/cifar10'