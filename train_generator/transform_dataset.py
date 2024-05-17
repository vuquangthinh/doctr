



print('Start generate data')

# get args
import argparse
import os
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

args = parser.parse_args()
  
import json

def convert_labels_txt_to_json(txt_file, json_file):
  labels = {}

  with open(txt_file, 'r') as file:
    for line in file:
      # Split the line by tab to separate filename and content
      parts = line.strip().split(' ', 1)
            
      # Extract the filename and content
      filename = parts[0]

      if len(parts) == 1:
        continue
            
      content = parts[1]

      # Add the filename and content to the labels dictionary
      labels[filename] = content

    with open(json_file, 'w') as file:
        json.dump(labels, file, indent=4, ensure_ascii=False)

convert_labels_txt_to_json(args.input_file, args.output_file)

print('End generate data')
