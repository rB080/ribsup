ACCOUNT="def-josedolz"
salloc --time=3:00:00 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=20G --account=${ACCOUNT}