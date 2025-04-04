# Development image with more bells and whistles
FROM shapepipe as shapepipe-dev


RUN apt-get update -y --quiet --fix-missing && \
    apt-get dist-upgrade -y --quiet --fix-missing && \
    apt-get install -y libgsl-dev

RUN pip install --no-cache-dir \ 
    snakemake

WORKDIR /sp_validation
COPY . /sp_validation

# Install shapepipe and symlink scripts
RUN pip install --no-cache-dir -e . && \ 