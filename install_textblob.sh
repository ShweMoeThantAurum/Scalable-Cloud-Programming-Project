#!/bin/bash
sudo pip3 install textblob numpy nltk
sudo python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/lib/python3.8/dist-packages/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/lib/python3.8/dist-packages/nltk_data')"
