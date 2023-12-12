#!/bin/bash

# Define the list of images
images=("1.jpeg" "2.jpeg" "3.jpeg" "4.jpeg" "5.jpeg" "6.jpeg" "7.jpeg" "8.jpeg" "9.jpeg" "10.jpeg" "11.jpeg" "12.jpeg" "13.jpeg")

# Define the rows_per_block value
rows_per_block=32

# Loop through each image and run tasks
for image in "${images[@]}"; do
    printf "${image} result\n"
    # Run task for CPU
    ./rbf "../images/$image" "../images/${image/.jpeg/_cpu.jpeg}" "$rows_per_block" 0

    # Run task for GPU Naive
    ./rbf "../images/$image" "../images/${image/.jpeg/_naive_gpu.jpeg}" "$rows_per_block" 1

    # Run task for GPU Refactor
    ./rbf "../images/$image" "../images/${image/.jpeg/_gpu.jpeg}" "$rows_per_block" 2
    
    printf "\n\n"
done
