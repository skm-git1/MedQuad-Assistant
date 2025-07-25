import os
import csv
import xml.etree.ElementTree as ET

# Define the root dataset directory
dataset_dir = "dataset"
output_csv_path = os.path.join(dataset_dir, "qa_dataset.csv")

# Initialize a list to hold all question-answer pairs
qa_data = []

# Traverse each subdirectory inside the dataset folder
for folder_name in os.listdir(dataset_dir):
    subfolder_path = os.path.join(dataset_dir, folder_name)
    if os.path.isdir(subfolder_path):
        # Iterate over each XML file in the subdirectory
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(subfolder_path, filename)
                try:
                    # Parse XML file
                    tree = ET.parse(file_path)
                    root = tree.getroot()

                    # Find all QAPairs
                    for qapairs in root.findall(".//QAPairs"):
                        for qapair in qapairs.findall("QAPair"):
                            question = qapair.findtext("Question", default="").strip()
                            answer = qapair.findtext("Answer", default="").strip()
                            if question and answer:
                                qa_data.append([question, answer])
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

# Save extracted data into a CSV file
with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Question", "Answer"])  # header
    writer.writerows(qa_data)

print(f"Extraction complete. CSV saved to: {output_csv_path}")
