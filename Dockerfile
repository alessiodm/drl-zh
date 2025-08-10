# Dockerfile.final
# ==============================================================================
# Final version: Re-adds essential EGL/OpenGL libraries for PyOpenGL to work.
# ==============================================================================
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Arguments for user ID, group ID, and code-server version.
ARG UID=1000
ARG GID=1000
ARG CODE_SERVER_VERSION=4.102.3

# Prevent apt-get from asking questions.
ENV DEBIAN_FRONTEND=noninteractive

# 💡 EGL FIX: Install the core EGL and OpenGL interface libraries.
# PyOpenGL needs these to find and use the EGL backend provided by the NVIDIA driver.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    curl \
    wget \
    tar \
    xz-utils \
    git \
    libgl1 \
    libegl1 && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Add conda to the system's PATH and set up shell activation.
ENV PATH="/opt/conda/bin:$PATH"

# Set the working directory for the application.
WORKDIR /app

# Accept Anaconda Terms of Service to allow non-interactive builds.
RUN conda tos accept

# Create the Conda environment with Python and necessary build tools.
#RUN conda create -p /opt/drl-env python=3.12.11 gxx_linux-64 swig -c conda-forge -y
RUN conda create -p /opt/drl-env python=3.12 gxx_linux-64 swig -c conda-forge -y

# Activate the Conda environment for all subsequent SHELL commands.
SHELL ["conda", "run", "-p", "/opt/drl-env", "/bin/bash", "-c"]

# Copy only the dependency definition files.
COPY poetry.lock pyproject.toml ./

# Set env var for Gymnasium ROM license.
ENV ACCEPT_ROM_LICENSE=YES

# Install Poetry and project dependencies, then clean up.
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    pip install poetry && \
    poetry install --no-root

# Clean up Conda caches.
RUN conda clean --all -y

# Install the Jupyter kernel using --prefix for system-wide access.
RUN python -m ipykernel install --prefix=/opt/conda --name "drl-zh-env" --display-name "Python (drl-zh)"

# Set the Matplotlib cache directory environment variable.
ENV MPLCONFIGDIR="/home/coder/.cache/matplotlib"

# Configure MuJoCo to use the headless EGL backend.
ENV MUJOCO_GL=egl

# Configure NVIDIA environment variables for GPU-accelerated graphics.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install code-server.
RUN curl -fL "https://github.com/coder/code-server/releases/download/v${CODE_SERVER_VERSION}/code-server-${CODE_SERVER_VERSION}-linux-amd64.tar.gz" \
    | tar -C /usr/local/lib -xz && \
    ln -s "/usr/local/lib/code-server-${CODE_SERVER_VERSION}-linux-amd64/bin/code-server" /usr/local/bin/code-server

# Create a non-root user 'coder' and grant sudo access.
RUN groupadd -g ${GID} coder && \
    useradd -m -s /bin/bash -u ${UID} -g ${GID} coder && \
    adduser coder sudo && \
    echo 'coder ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Pre-create config/cache directories and set permissions.
RUN mkdir -p /home/coder/.local/share/code-server/User /home/coder/.cache && \
    echo '{ "python.defaultInterpreterPath": "/opt/drl-env/bin/python" }' > /home/coder/.local/share/code-server/User/settings.json && \
    chown -R ${UID}:${GID} /home/coder/.local /home/coder/.cache

# Switch to the non-root user for all subsequent user-specific setup.
USER coder

# Initialize Conda for the coder user's shell and set default environment.
RUN conda init bash && \
    echo "conda activate /opt/drl-env" >> /home/coder/.bashrc

# Install extensions as the 'coder' user directly.
RUN code-server --install-extension ms-python.black-formatter

# Set the final working directory for the user.
WORKDIR /home/coder/project

# Expose the port for code-server.
EXPOSE 8080

# This forces the ENTRYPOINT to run inside the activated env, making the
# kernel visible to VS Code's Jupyter extension on startup.
SHELL ["conda", "run", "-p", "/opt/drl-env", "--no-capture-output", "/bin/bash", "-c"]

# Set the container's entrypoint.
ENTRYPOINT ["code-server", "--auth", "none", "--bind-addr", "0.0.0.0:8080", "."]