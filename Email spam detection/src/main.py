import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

import logging

import json

import tkinter as tk
from tkinter import simpledialog, messagebox, ttk

# Configure logging
logging.basicConfig(filename='model_log.txt', level=logging.INFO)

# Reading CSV dataset
data_frame = pd.read_csv('data_set.csv')

# Dropping rows with missing values
data_frame = data_frame.dropna()

# Printing content
logging.info(data_frame)

# Printing content
print(data_frame)

# Replacing label column in the dataset from 'spam' and 'ham' to 0 and 1 respectively
data_frame['Label'] = data_frame['Label'].map({'spam': 0, 'ham': 1})

# Counting all the rows in the label column
label_counts = data_frame['Label'].value_counts()

# Number of ham and spam emails in dataset
print("Number of ham emails: {}".format(label_counts[1]))
print("Number of spam emails: {}".format(label_counts[0]))

# Splitting dataset into training and testing, 80% and 20% respectively
x_train, x_test, y_train, y_test = train_test_split(data_frame['Content'], data_frame['Label'], test_size=0.2)

# Converting the textual content of the emails into a format that can
# be used as input for a machine-learning model allows the model to work
# with numerical features rather than raw text

# Transforming data into feature vectors using TF-IDF vectorization
# NOTE: min_df: A word should be present in at least 1 document to be considered
# NOTE: stop_words: Common words that are often irrelevant for NLP tasks
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# Converting labels to integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Training ML Model using Logistic Regression
model = LogisticRegression()
model.fit(x_train_features, y_train)

# Making predictions on the training set and calculating the accuracy of the model
prediction_train = model.predict(x_train_features)
accuracy_train = accuracy_score(y_train, prediction_train)
print('Model Accuracy (training): {}%'.format(int(accuracy_train * 100)))

logging.info('Model Accuracy (training): {}%'.format(int(accuracy_train * 100)))

prediction_test = model.predict(x_test_features)
accuracy_test = accuracy_score(y_test, prediction_test)
print('Model Accuracy (testing): {}%'.format(int(accuracy_train * 100)))

logging.info('Model Accuracy (testing): {}%'.format(int(accuracy_train * 100)))

print("")

# Precision, recall, F1-score and Support
print("Classification Report:")
print("-" * 25)
print(classification_report(y_test, prediction_test, target_names=['ham', 'spam']))

logging.info("Classification Report:")
logging.info("-" * 25)
logging.info(classification_report(y_test, prediction_test, target_names=['ham', 'spam']))

# Empty arrays to store inbox and spam folder emails
inbox = []
spam_folder = []

# Array of emails
incoming_emails = [
    "Dear user, your account needs verification. Click here to proceed.",
    "This email contains your exclusive offers",
    "Please verify your account by clicking on the following link.",
    "Hi, I hope this email finds you well. Let's catch up soon!",
    "Congratulations! You've won a free vacation. Click to claim your prize.",
    "Dear Traveler, Thank you for your continued support for Genshin Impact! We've always placed great importance on your experiences and impressions of traveling around Teyvat. For this reason, we've specially designed this survey to understand more about various different aspects of your recent experience playing Genshin Impact. Click on the button below to fill in the survey and share your feedback! The deadline for filling in this survey is 12/31/2023 23:59 (UTC+8). The survey is only available for you to fill out. Please do not share the contents of this mail or the survey link with others. Thank you again for your support for Genshin Impact! We hope you have fun in Teyvat!",
    "You can choose the 30-minute session that best fits your needs. Get help with setting up your new iPhone, including unboxing, signing in and initiating data transfer. Or, we can get you started with navigating iPhone and share simple ways to stay connected to what matters most. We can also explore the latest features and Apple apps that make everyday tasks easier. Whether you’re new to iPhone or need a refresher, we’ll help you get started and discover new favourites.",
    "Save yourself the stress of holiday shopping and grab an ILLVZN Gift Card. Available at £50, £100 and £200 values. Happy Holidays from the ILLVZN Fam. See More, Say Less™️"
]


# Function to classify and add emails to the correct array
def classifying_emails(emails):
    global inbox
    global spam_folder

    for email_content in emails:
        # Checking if the email doesn't already exist in the arrays
        if email_content not in inbox and email_content not in spam_folder:
            # Transforming email content into a format readable by the ML model
            email_features = feature_extraction.transform([email_content])

            # Making a prediction using the model
            predicted_label = model.predict(email_features)[0]

            # Mapping 0 to spam and 1 to ham
            predicted_label = 'spam' if predicted_label == 0 else 'ham'

            # Adding the email to the correct array
            if predicted_label == 'ham':
                inbox.append(email_content)
            else:
                spam_folder.append(email_content)
        else:
            print("Email already exist.")


# Function to move email from inbox to spam folder
def inbox_to_spam():
    global inbox
    global spam_folder
    try:
        email_number = int(input("Enter the number of the email to move to the spam folder: "))
        if 1 <= email_number <= len(inbox):
            moved_email = inbox.pop(email_number - 1)
            spam_folder.append(moved_email)
            print("Email has been moved to the spam folder.")
        else:
            print("Invalid email number.")
    except ValueError:
        print("Invalid input.")


# Function to move email from spam folder to inbox
def spam_to_inbox():
    global inbox
    global spam_folder
    try:
        mail_number = int(input("Enter the number of the email to move to the spam folder: "))
        if 1 <= mail_number <= len(spam_folder):
            moved_mail = spam_folder.pop(mail_number - 1)
            inbox.append(moved_mail)
            print("Email has been moved to your inbox.")
        else:
            print("Invalid email number.")
    except ValueError:
        print("Invalid input.")


classifying_emails(incoming_emails)

while True:
    try:
        print("\nPlease select one of the following:")
        print("-----------------------------------------")
        print("| 1. Open inbox                         |")
        print("| 2. Open spam folder                   |")
        print("| 3. Quit                               |")
        print("-----------------------------------------")
        choice = input("Choice: ")

        if choice == '1':
            # Emails in inbox
            print("\nInbox:")
            print("-" * 100)
            for i, email in enumerate(inbox, 1):
                print(f"{i}. {email}")
                print("-" * 100)

            # In case email was misclassified:
            print("\nAdditional Options:")
            sub_choice = input("Enter 1 to move an email from your inbox to the spam folder or any key to go back: ")

            if sub_choice == '1':
                inbox_to_spam()

        elif choice == '2':
            # Emails in spam folder
            print("\nSpam folder:")
            print("-" * 100)
            for i, spam in enumerate(spam_folder, 1):
                print(f"{i}. {spam}")
                print("-" * 100)

            # In case email was misclassified:
            print("\nAdditional Options:")
            sub_choice2 = input("Enter 1 to move an email from your spam folder to inbox or any key to go back: ")

            if sub_choice2 == '1':
                spam_to_inbox()

        elif choice == '3':
            print("Program has ended.")
            break
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")
    except Exception as e:
        print(f"Error: {e}")


# GUI:
# Function to classify and add emails to the correct array
def classifying_emails(emails):
    global inbox
    global spam_folder

    for email_content in emails:
        # Checking if the email doesn't already exist in the arrays
        if email_content not in inbox and email_content not in spam_folder:
            # Transforming email content into a format readable by the ML model
            email_features = feature_extraction.transform([email_content])

            # Making a prediction using the model
            predicted_label = model.predict(email_features)[0]

            # Mapping 0 to spam and 1 to ham
            predicted_label = 'spam' if predicted_label == 0 else 'ham'

            # Adding the email to the correct array
            if predicted_label == 'ham':
                inbox.append(email_content)
            else:
                spam_folder.append(email_content)
        else:
            print("Email already exists.")


# Function to move email from inbox to spam folder
def inbox_to_spam():
    global inbox
    global spam_folder
    try:
        email_number = simpledialog.askinteger("Move email to spam folder",
                                               "Enter the number of the email to move to the spam folder:")
        if email_number is not None and 1 <= email_number <= len(inbox):
            moved_email = inbox.pop(email_number - 1)
            spam_folder.append(moved_email)
            update_gui_lists()
            messagebox.showinfo("Success", "Email has been moved to the spam folder.")
        else:
            messagebox.showerror("Error", "Invalid email number.")
    except ValueError:
        messagebox.showerror("Error", "Invalid input.")


# Function to move email from spam folder to inbox
def spam_to_inbox():
    global inbox
    global spam_folder
    try:
        email_number = simpledialog.askinteger("Move email to inbox",
                                               "Enter the number of the email to move to the inbox:")
        if email_number is not None and 1 <= email_number <= len(spam_folder):
            moved_email = spam_folder.pop(email_number - 1)
            inbox.append(moved_email)
            update_gui_lists()
            messagebox.showinfo("Success", "Email has been moved to your inbox.")
        else:
            messagebox.showerror("Error", "Invalid email number.")
    except ValueError:
        messagebox.showerror("Error", "Invalid input.")


# Function to update emails in GUI
def update_gui_lists():
    inbox_listbox.delete(0, tk.END)
    spam_listbox.delete(0, tk.END)

    # Populate Inbox Listbox
    for i, email in enumerate(inbox, 1):
        inbox_listbox.insert(tk.END, f"{i}. {email}")

    # Populate Spam Folder Listbox
    for i, spam in enumerate(spam_folder, 1):
        spam_listbox.insert(tk.END, f"{i}. {spam}")


# Function to exit
def exit_program():
    root.destroy()


# GUI
root = tk.Tk()
root.title("Model simulation in real life")
root.geometry("1500x800")

# Frames
# Left frame
left_frame = ttk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=2)

# Right frame
right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=1, pady=2)

# Label for Inbox
inbox_label = ttk.Label(left_frame, text="Inbox:")
inbox_label.pack()

# Listbox to display Inbox emails
inbox_listbox = tk.Listbox(left_frame)
inbox_listbox.pack(expand=True, fill='both')

# Label for Spam Folder
spam_label = ttk.Label(left_frame, text="Spam Folder:")
spam_label.pack()

# Listbox to display Spam Folder emails
spam_listbox = tk.Listbox(left_frame)
spam_listbox.pack(expand=True, fill='both')

# Buttons
inbox_to_spam_button = ttk.Button(right_frame, text="Move to Spam", command=inbox_to_spam)
inbox_to_spam_button.pack(pady=10)

spam_to_inbox_button = ttk.Button(right_frame, text="Move to Inbox", command=spam_to_inbox)
spam_to_inbox_button.pack(pady=10)

exit_button = ttk.Button(right_frame, text="Exit", command=exit_program)
exit_button.pack(pady=10)

# Classifying emails
classifying_emails(incoming_emails)
update_gui_lists()

# Running the GUI
root.mainloop()

# Testing Model
prediction_test = model.predict(x_test_features)

results_df = pd.DataFrame({
    'Expected Output': y_test.values,
    'Actual Output': prediction_test
})

results_df['Expected Output'] = results_df['Expected Output'].map({0: 'spam', 1: 'ham'})
results_df['Actual Output'] = results_df['Actual Output'].map({0: 'spam', 1: 'ham'})

print("Comparison of Expected vs Actual Output with Email Content:")
print("-" * 60)

logging.info("Comparison of Expected vs Actual Output with Email Content:")
logging.info("-" * 60)

# Displaying emails which were misclassified
# Using zip function to compare 2 lists
misclassified_count = 0
for index, (expected, actual) in enumerate(zip(y_test.values, prediction_test)):
    if expected != actual:
        misclassified_count += 1
        email_content = data_frame.iloc[y_test.index[index]]['Content']
        print(f"Expected: {'Spam' if expected == 0 else 'Ham'}, Actual: {'Spam' if actual == 0 else 'Ham'}")
        print("Email Content:")
        print(email_content)
        print("-" * 80)

        logging.info(f"Expected: {'Spam' if expected == 0 else 'Ham'}, Actual: {'Spam' if actual == 0 else 'Ham'}")
        logging.info("Email Content:")
        logging.info(email_content)
        logging.info("-" * 80)

print(f"Total number of misclassified emails: {misclassified_count}")
print("")
print("Results on Testing Data:")
print("")
print(results_df)

logging.info(f"Total number of misclassified emails: {misclassified_count}")
logging.info("")
logging.info("Results on Testing Data:")
logging.info("")
logging.info(results_df)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, prediction_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['spam', 'ham'], yticklabels=['spam', 'ham'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()