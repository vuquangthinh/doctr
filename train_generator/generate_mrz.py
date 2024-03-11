# import csv
# from faker import Faker

# fake = Faker()

# def calculate_check_digit(mrz):
#     """Calculates the check digit for the MRZ string."""
#     weights = [7, 3, 1]
#     total = 0
#     for i, char in enumerate(mrz):
#         if char.isdigit():
#             total += int(char) * weights[i % 3]
#     return str(total % 10)

# def generate_mrz_data(num_samples):
#     data = []
#     for _ in range(num_samples):
#         document_type = fake.random_element(['P', 'I', 'V', 'C'])
#         country_code = fake.country_code()
#         surname = fake.last_name().upper()
#         given_names = fake.first_name().upper()
#         document_number = fake.random_number(digits=9)
#         nationality = fake.country_code()
#         date_of_birth = fake.date_of_birth(minimum_age=18).strftime('%y%m%d')
#         sex = fake.random_element(['M', 'F'])
#         expiration_date = fake.future_date(end_date='+10y').strftime('%y%m%d')
#         personal_number = fake.random_number(digits=7)
#         check_digit = calculate_check_digit(f"{document_number}{date_of_birth}{expiration_date}{personal_number}")
#         mrz = f"{document_type}{country_code}{surname}<<{given_names}<<{document_number}{check_digit}{nationality}{date_of_birth}{sex}{expiration_date}{personal_number}{check_digit}"
#         data.append([mrz])
#     return data

# # Define the number of samples to generate
# num_samples = 1000

# # Generate MRZ dataset
# mrz_data = generate_mrz_data(num_samples)

# # Save dataset to a CSV file
filename = '/home/thanhpcc/Documents/testx/doctr/train_generator/mrz.txt'
# with open(filename, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(mrz_data)

# print(f"MRZ dataset with {num_samples} samples saved to {filename}.")

import random
import string

def generate_random_string(length):
    characters = string.ascii_uppercase + string.digits + "<"
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# Generate a batch of 1000 random strings
batch_size = 100000
# random_strings = [generate_random_string(12) for _ in range(batch_size)]

# # Print the batch of random strings
# for random_string in random_strings:
#     print(random_string)

with open(filename, 'w') as file:
    for _ in range(batch_size):
        file.write(generate_random_string(31) + "\n")