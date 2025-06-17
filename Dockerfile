# Use Python as the base image to ensure that the version is consistent with the project requirements
FROM python:3.10.16-bookworm

# Whether to use a mirror source for PyPI (True means use TUNA mirror)
# GFW is used to determine whether to switch to the TUNA mirror in China
ARG GFW=false

# Set working directory
WORKDIR /app

# Copy project files to the working directory
COPY . .

# Install git-lfs to avoid problems with large files
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs
RUN git lfs install

# Install dependencies
RUN if [ "$GFW" = "True" ]; then \
        echo "Using TUNA mirror for PyPI"; \
        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    else \
        echo "Using default PyPI"; \
        pip install -r requirements.txt; \
    fi

# Conditionally set Hugging Face mirror for runtime if GFW is True
RUN if [ "$GFW" = "true" ]; then \
        echo "Setting Hugging Face mirror to https://hf-mirror.com/ for runtime"; \
        echo "export HF_ENDPOINT=https://hf-mirror.com/" >> /etc/profile; \
    fi

# Set Hugging Face mirror for runtime (not only during Python download.py)
# This ensures HF_ENDPOINT is set based on the GFW condition
ENV HF_ENDPOINT="https://hf-mirror.com/"

# Download models
RUN python ./download.py

# Exposed port
EXPOSE 5003

# Run the startup script when starting the container
CMD ["sh", "-c", "python -u src/model_server/server.py"]