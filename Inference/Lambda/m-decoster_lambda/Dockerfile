# Define custom function directory
ARG FUNCTION_DIR="/function/"

FROM python:3.9 as build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev \
  libgl1
# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY app/* ${FUNCTION_DIR}

WORKDIR ${FUNCTION_DIR}

RUN pip install -r requirements.txt --target ${FUNCTION_DIR}

# Install the function's dependencies
RUN pip install \
    --target ${FUNCTION_DIR} \
        awslambdaric

FROM python:3.9

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# https://github.com/mitchellharper12/lambda-pthread-nameshim
COPY lambda-pthread-nameshim/* ${FUNCTION_DIR}
RUN make pthread_shim.so && cp pthread_shim.so /opt
# Download models
RUN python init_mp_resnet.py

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD ["app.lambda_handler"]
