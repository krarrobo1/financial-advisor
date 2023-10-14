import os
import shutil

source_directory = "nasdaq_annual_reports"

destination_directory = "documents"


# Function to copy PDF files
def copy_pdf_files(source, destination):
    for root_folder, _, files in os.walk(source):
        for file in files:
            if file.endswith(".pdf"):
                source_file = os.path.join(root_folder, file)
                destination_file = os.path.join(destination, file)
                shutil.copy(source_file, destination_file)
                print(f"Copying {source_file} to {destination_file}")


# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Copy PDF files
copy_pdf_files(source_directory, destination_directory)

print("PDF file copy completed.")
