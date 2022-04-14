conda activate spot_ros
echo "Killing all tmux sessions..."
tmux kill-server
sleep 2
echo "Starting roscore tmux..."
tmux new -s roscore -d '$CONDA_PREFIX/bin/roscore'
sleep 1
echo "Starting other tmux nodes"
tmux new -s headless_estop -d '$CONDA_PREFIX/bin/python -m spot_wrapper.headless_estop'
tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.spot_ros_node'
tmux new -s propio_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.spot_ros_node -p'
tmux new -s tts_sub -d '$CONDA_PREFIX/bin/python -m spot_rl.spot_ros_node -t'
sleep 3
tmux ls
