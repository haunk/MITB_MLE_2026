# Based on lab 5 Dockerfile, updated to a newer Airflow image
# (Airflow 2.10.5 on Debian 12 "bookworm", Python 3.12 = same Python as assignment 1)
FROM apache/airflow:2.10.5-python3.12

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless) and procps (for 'ps'); bash is already in the image
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # On arm64 machines (e.g. Apple Silicon) Java is installed as java-17-openjdk-arm64,
    # so link it to the amd64 path used by JAVA_HOME below
    if [ ! -d /usr/lib/jvm/java-17-openjdk-amd64 ]; then \
        ln -s /usr/lib/jvm/java-17-openjdk-* /usr/lib/jvm/java-17-openjdk-amd64; \
    fi

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
