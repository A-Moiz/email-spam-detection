import os
import csv

# Directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path of email folder
email_folder_path = os.path.join(base_dir, '../emails')

# Output CSV File path
file_location = os.path.dirname(os.path.abspath(__file__))
dataSet_csv = os.path.join(file_location, 'data_set.csv')

# List to store data
data = []

# Counter for numbering emails
email_number = 1

try:
    # Iterating through each sub-folder (ham and spam)
    for label in ['ham', 'spam']:
        label_folder_path = os.path.join(email_folder_path, label)

        # Iterating through each text file in the sub-folders
        for filename in os.listdir(label_folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(label_folder_path, filename)

                try:
                    # Reading content of text file
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()

                    # Adding content to List
                    data.append([email_number, content, label])

                    # Incrementing email number
                    email_number += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

# Exceptions
except FileNotFoundError:
    print(f"Error: The specified folder '{email_folder_path} does not exist.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

else:
    try:
        with open(dataSet_csv, 'w', newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Writing header
            csv_writer.writerow(['Number', 'Content', 'Label'])

            # Writing data rows
            csv_writer.writerows(data)

        print(f"CSV File created at {dataSet_csv}")

    except Exception as e:
        print(f"Error writing to CSV File: {e}")