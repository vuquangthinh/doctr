

# import os
# import shutil

# def create_directory(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#         print(f"Directory '{directory}' created.")
#     else:
#         print(f"Directory '{directory}' already exists.")

# # Example usage

# def copy_files(source_dir, destination_dir):
#     # Get the list of files in the source directory
#     files = os.listdir(source_dir)

#     # Iterate through each file in the source directory
#     for file_name in files:
#         # Construct the full path of the source file
#         source_file = os.path.join(source_dir, file_name)

#         # Construct the full path of the destination file
#         destination_file = os.path.join(destination_dir, file_name)

#         # Copy the file to the destination directory
#         shutil.copy2(source_file, destination_file)

# # Example usage
# source_directory = '/Users/vuquangthinh/Documents/fencilux/eKYC/xyz/vocr/out'
# destination_directory = './data/val_data'

# create_directory(destination_directory + '/images')

# copy_files(source_directory, destination_directory + '/images')


# import json

# def convert_labels_txt_to_json(txt_file, json_file):
#     labels = {}

#     with open(txt_file, 'r') as file:
#         for line in file:
#             # Split the line by tab to separate filename and content
#             parts = line.strip().split(' ')
            
#             # Extract the filename and content
#             filename = parts[0]

#             if len(parts) == 1:
#                 continue
            
#             content = parts[1]

#             # Add the filename and content to the labels dictionary
#             labels[filename] = content

#     with open(json_file, 'w') as file:
#         json.dump(labels, file, indent=4, ensure_ascii=False)

# # Example usage
# txt_file_path = source_directory + '/labels.txt'
# json_file_path = destination_directory + '/labels.json'

# convert_labels_txt_to_json(txt_file_path, json_file_path)
def get_characters_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the contents of the file
        text = file.read()

    # Create an empty set to store unique characters
    unique_chars = set()

    # Iterate over each character in the text
    for char in text:
        # Add the character to the set
        unique_chars.add(char)

    return unique_chars

# Example usage
file_path = '/Users/vuquangthinh/Documents/fencilux/eKYC/xyz/vocr/x.txt'
characters = get_characters_from_file(file_path)
print(''.join(characters))


# # Example usage
# sentence = "This is a sample sentence."
# characters = get_unique_characters(sentence)
# print(''.join(characters))