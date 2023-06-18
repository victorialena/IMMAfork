!/bin/bash -ex

env=bball
dim=4
directory='logs/bball/06-03-23_exp1'

python eval.py --env $env --input_size $dim --output_dir $directory


env=springs5
dim=4
directory='logs/springs5/05-31-23_exp2'

python eval.py --env $env --input_size $dim --output_dir $directory


# env=motion
# dim=6
# directory='logs/motion/06-12-23_exp3'

# python eval.py --env $env --input_size $dim --output_dir $directory