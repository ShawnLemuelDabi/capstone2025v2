import csv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def extract_urls_from_csv(input_csv_path, output_dir):
    """
    Extracts URLs from the 'video_url' column of a CSV file and saves them to a text file
    within an output directory named after the CSV file.

    Args:
        input_csv_path (str): The full path to the input CSV file.
        output_dir (str): The base directory where the output directory will be created.
    """
    urls = []
    try:
        with open(input_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'video_url' not in reader.fieldnames:
                print(f"Error: The CSV file does not contain a 'video_url' column.")
                return

            for row in reader:
                url = row.get('video_url')
                if url:
                    urls.append(url)

        # Get the filename (without extension) from the input path
        base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
        output_sub_dir = os.path.join(output_dir, base_name)
        os.makedirs(output_sub_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        output_txt_path = os.path.join(output_sub_dir, f"{base_name}_urls.txt")

        with open(output_txt_path, 'w', encoding='utf-8') as txtfile:
            txtfile.write('[\n')
            for i, url in enumerate(urls):
                txtfile.write(f'  "{url}"')
                if i < len(urls) - 1:
                    txtfile.write(',')
                txtfile.write('\n')
            txtfile.write(']\n')

        print(f"Successfully extracted {len(urls)} URLs and saved them to: {output_txt_path}")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at: {input_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Initialize Tkinter (but hide the main window)
    root = Tk()
    root.withdraw()

    # Set the initial directory for the file dialog
    initial_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs"

    # Open a file dialog to select the input CSV file
    input_file_path = askopenfilename(
        title="Select Input CSV File",
        initialdir=initial_dir,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if input_file_path:
        output_base_dir = initial_dir  # The main output directory
        extract_urls_from_csv(input_file_path, output_base_dir)
    else:
        print("No input CSV file selected.")