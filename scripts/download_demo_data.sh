#!/bin/bash
mkdir -p data
cd data
echo "Downloading demo data..."
wget https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/data/demo_data.zip
unzip demo_data.zip
echo "Done!"