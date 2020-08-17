#!/bin/bash
mkdir -p data
cd data
echo "Start downloading ..."
wget https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/data/synthetic_room_dataset.zip
unzip synthetic_room_dataset.zip
echo "Done!"