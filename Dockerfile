# XeLaTeX Compilation Environment for ΨQRH Hamiltonian Monte Carlo Paper
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    texlive-xetex \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-bibtex-extra \
    biber \
    python3 \
    python3-pip \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for any preprocessing
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    scipy \
    torch

# Set working directory
WORKDIR /workspace

# Copy paper files
COPY monte-carlo-psiqrh-paper.tex /workspace/
COPY references.bib /workspace/
COPY Visualizing_2D_HamiltonianDynamics.png /workspace/
COPY Visualizing_1D_HamiltonianDynamics.png /workspace/
COPY VisualizingSampleDistributions.png /workspace/

# Default command
CMD ["bash"]