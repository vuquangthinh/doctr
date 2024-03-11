khmer_numerals = {
    0: '០',
    1: '១',
    2: '២',
    3: '៣',
    4: '៤',
    5: '៥',
    6: '៦',
    7: '៧',
    8: '៨',
    9: '៩'
}

def convert_to_khmer_numerals(number):
    khmer_numeral = ''
    for digit in str(number):
        if digit.isdigit():
            khmer_numeral += khmer_numerals[int(digit)]
        else:
            khmer_numeral += digit
    return khmer_numeral
    
import random

with open('./dates.txt', 'w') as d:
  # Generate 1000 random dates
  for _ in range(1000):
      # Get a random day between 1 and 31
      random_day = random.randint(1, 31)

      # Get a random month between 1 and 12
      random_month = random.randint(1, 12)

      # Get a random year between 1900 and 2100
      random_year = random.randint(1900, 2100)

      # Format the date as "dd/mm/yyyy"
      date = f"{random_day:02d}.{random_month:02d}.{random_year}"

      # Print the random date
      print(convert_to_khmer_numerals(date))
      d.write(convert_to_khmer_numerals(date) + '\n')
    
