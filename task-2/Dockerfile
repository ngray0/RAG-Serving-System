FROM python:3.10

WORKDIR /teamspace/studios/this_studio/MLS-G31/task-2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir cupy-cuda12x  
#RUN pip install --no-cache-dir cupy-cuda11x  # For CUDA 11.x

# Pre-download models
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct'); \
    AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    from transformers import pipeline; \
    pipeline('text-generation', model='Qwen/Qwen2.5-1.5B-Instruct')"
    

COPY . .

ENV PYTHONPATH=/task-2

# env variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MAX_BATCH_SIZE=16
ENV MAX_WAIT_TIME=1
ENV DEVICE=cuda

# Expose the port the app runs on
EXPOSE 8000

# start
CMD ["python", "main.py"]