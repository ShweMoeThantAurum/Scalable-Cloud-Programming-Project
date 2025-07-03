#!/bin/bash

# Exit on any error
set -e

# Log file for debugging
LOG_FILE="/tmp/bootstrap.log"
echo "Starting bootstrap script..." | tee -a $LOG_FILE

# Update system packages
echo "Updating system packages..." | tee -a $LOG_FILE
sudo yum update -y >> $LOG_FILE 2>&1

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
NLTK_DATA_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/nltk_data"
echo "Detected Python version: $PYTHON_VERSION" | tee -a $LOG_FILE
echo "NLTK data path: $NLTK_DATA_PATH" | tee -a $LOG_FILE

# Install Python dependencies
echo "Installing Python dependencies..." | tee -a $LOG_FILE
sudo pip3 install boto3 pandas plotly kaleido textblob numpy nltk >> $LOG_FILE 2>&1

# Install NLTK data for TextBlob
echo "Installing NLTK data..." | tee -a $LOG_FILE
sudo mkdir -p $NLTK_DATA_PATH
sudo python3 -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA_PATH'); nltk.download('averaged_perceptron_tagger', download_dir='$NLTK_DATA_PATH'); nltk.download('brown', download_dir='$NLTK_DATA_PATH')" >> $LOG_FILE 2>&1

# Install Google Chrome for Kaleido (Plotly image rendering)
echo "Installing Google Chrome..." | tee -a $LOG_FILE
sudo wget https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm >> $LOG_FILE 2>&1
sudo yum install -y ./google-chrome-stable_current_x86_64.rpm >> $LOG_FILE 2>&1
sudo rm -f google-chrome-stable_current_x86_64.rpm >> $LOG_FILE 2>&1

# Verify installations
echo "Verifying Python package installations..." | tee -a $LOG_FILE
pip3 show boto3 pandas plotly kaleido textblob numpy nltk >> $LOG_FILE 2>&1

# Check NLTK data
echo "Verifying NLTK data..." | tee -a $LOG_FILE
ls -l $NLTK_DATA_PATH >> $LOG_FILE 2>&1

echo "Bootstrap script completed successfully." | tee -a $LOG_FILE
exit 0
