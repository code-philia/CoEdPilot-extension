# Use Python as the base image to ensure that the version is consistent with the project requirements
FROM python:3.10.16-bookworm

# Whether to use a mirror source
# If it is empty, do not use the mirror source
# otherwise, use this value as the mirror source for installing dependencies
ARG MIRROR_SOURCE 

# Choose language for your project, defalut language is python
ARG LANG=python

# Set working directory
WORKDIR /app

# Copy project files to the working directory
COPY . .

# Install git-lfs to avoid problems with large files
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs
RUN git lfs install

# Install dependencies
RUN if [ -n "$MIRROR_SOURCE" ]; then \
        echo "Using mirror source: $MIRROR_SOURCE"; \
        pip install -r requirements.txt -i "$MIRROR_SOURCE"; \
    else \
        echo "Using default PyPI"; \
        pip install -r requirements.txt; \
    fi

# Download models
RUN python ./download.py

# Exposed port
EXPOSE 5003

# Run the startup script when starting the container
CMD ["sh", "-c", "python -u src/model_server/server.py"]
