DOCKER_IMAGE=$(<version.txt)

docker run \
    -it --gpus all \
    --name=ogr_cmu_${USER} \
    -e DISPLAY=${DISPLAY} \
    --net=host \
    --ipc=host \
    -v ${HOME}/.Xauthority:/root/.Xauthority:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/../src:/ogr_cmu/src \
    -v $(pwd)/../data/:/ogr_cmu/data/ \
    -v $(pwd)/../output:/ogr_cmu/output \
    -v $(pwd)/../scripts:/ogr_cmu/scripts \
    -v $(pwd)/../.vscode:/ogr_cmu/.vscode \
    -v $(pwd)/../models:/ogr_cmu/models \
${DOCKER_IMAGE}