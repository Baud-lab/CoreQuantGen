FROM rocker/r-ver:4.3.2

ENV DEBIAN_FRONTEND=noninteractive

# System libraries needed to compile rhdf5 and its dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    zlib1g-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CRAN packages: MASS ships with base R already; only optparse and matrixcalc are installed here
RUN Rscript -e 'install.packages(c("optparse", "matrixcalc"))'

# Bioconductor package: rhdf5 requires BiocManager, not install.packages()
RUN Rscript -e 'install.packages("BiocManager"); BiocManager::install("rhdf5", update=FALSE, ask=FALSE)'

# Quick sanity check baked into the build: fails the build immediately if anything didn't install
RUN Rscript -e 'suppressMessages(library("MASS")); suppressMessages(library("rhdf5")); suppressMessages(library("optparse")); suppressMessages(library("matrixcalc")); cat("All packages loaded OK\n")'
