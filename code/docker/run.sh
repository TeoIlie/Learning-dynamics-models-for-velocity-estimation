docker rm learning_through_kalman_filter
xhost + local:root

docker run -it \
    --gpus all \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    --volume="$(pwd):/learning_through_kalman_filter" \
    --privileged \
    --network=host \
    --name=learning_through_kalman_filter \
    learning_through_kalman_filter
