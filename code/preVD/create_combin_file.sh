#!/bin/bash

# Get the starting number from the first argument
start_num=$1
# Calculate the end number 
end_num=$2
# Loop from start_num to end_num (inclusive) with a step size of 2
for (( i=$start_num; i<=$end_num; i+=2 ))
do
# Print the current pair with a newline
#echo "$i $((i+1))"
echo "$i,$((i+1))" # this gives a real csv
done 
