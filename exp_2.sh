

TaskList="1"
for task in $TaskList
do
  echo "$task"
#  CUDA_VISIBLE_DEVICES=5 python exp_train_clean_model.py --exp="$task"
#
#  CUDA_VISIBLE_DEVICES=5 python trigger_generation.py --exp="$task"
#
#  CUDA_VISIBLE_DEVICES=5 python exp_inject_backdoor.py --exp="$task" --type=0
CUDA_VISIBLE_DEVICES=5 python exp_inject_backdoor.py --exp="$task" --type=1

#  CUDA_VISIBLE_DEVICES=5 python exp_inject_baseline.py --exp="$task" --baseline=0
#  CUDA_VISIBLE_DEVICES=5 python exp_inject_baseline.py --exp="$task" --baseline=1
done