conda activate spot_ros
echo "Killing all tmux sessions..."
tmux kill-server
sleep 2
echo "Starting roscore"
tmux new -s roscore -d '/home/spot/miniconda3/envs/spot_ros/bin/roscore'
sleep 3
echo "Starting nodes"
tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.spot_ros_node'
tmux new -s propio_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.spot_ros_node -p'
tmux new -s tts_sub -d '$CONDA_PREFIX/bin/python -m spot_rl.spot_ros_node -t'
