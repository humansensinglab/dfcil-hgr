DOCKER_IMAGE=$(<version.txt)
cd ..

docker build \
-f docker/Dockerfile \
-t ${DOCKER_IMAGE} \
--build-arg USER_ID=$(id -u) \
--build-arg GROUP_ID=$(id -g) .

