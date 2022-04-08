conda activate spot_ros
echo "Killing all tmux sessions..."
tmux kill-server
sleep 2
echo "Starting roscore"
tmux new -s roscore -d '/home/spot/miniconda3/envs/spot_ros/bin/roscore'
sleep 3
echo "Starting nodes"
tmux new -s img_pub -d '$CONDA_PREFIX/envs/spot_ros/bin/python /home/spot/pvp/spot_rl_experiments/spot_ros_node.py'
tmux new -s propio_pub -d '$CONDA_PREFIX/envs/spot_ros/bin/python /home/spot/pvp/spot_rl_experiments/spot_ros_node.py -p'
tmux new -s tts_sub -d '$CONDA_PREFIX/envs/spot_ros/bin/python /home/spot/pvp/spot_rl_experiments/spot_ros_node.py -t'
