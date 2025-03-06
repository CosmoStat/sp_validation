# Development image with more bells and whistles
FROM shapepipe as shapepipe-dev

RUN pip install --no-cache-dir 
    snakemake