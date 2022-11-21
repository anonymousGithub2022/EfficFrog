python main.py \
  --arch RANet \
  --gpu '0,1,2,3' \
  --data-root {your data root} \
  --data 'ImageNet' \
  --growthRate 16 \
  --step 8 \
  --stepmode 'even' \
  --scale-list '1-2-3-4' \
  --grFactor '4-2-1-1' \
  --bnFactor '4-2-1-1'