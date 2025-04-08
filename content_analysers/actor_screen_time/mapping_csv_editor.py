import os
import tkinter as tk
from tkinter import filedialog, messagebox
import csv


def generate_csv(input_dir, output_dir, data_type):
    # Get all jpg files in the input directory
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

    # Sort files by their numerical value (frame0.jpg, frame1.jpg, etc.)
    jpg_files.sort(key=lambda x: int(x[5:-4]))

    # Determine output filename and class value
    output_filename = "negatives_mapping.csv" if data_type == "negative" else "positives_mapping.csv"
    class_value = 0 if data_type == "negative" else 1

    # Create the output path
    output_path = os.path.join(output_dir, output_filename)

    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image_ID', 'Class'])
        for file in jpg_files:
            writer.writerow([file, class_value])

    messagebox.showinfo("Success", f"{output_filename} has been generated successfully at {output_path}")


def select_directories_and_generate(data_type):
    # Set default input directory
    default_input_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\content_analysers\actor_screen_time\model_trainer"

    # Get script directory for default output
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Open directory dialogs
    input_dir = filedialog.askdirectory(
        initialdir=default_input_dir,
        title="Select Input Directory with Frames"
    )

    if not input_dir:  # If user cancelled
        return

    output_dir = filedialog.askdirectory(
        initialdir=script_dir,
        title="Select Output Directory for CSV",
        mustexist=True
    )

    if not output_dir:  # If user cancelled
        return

    generate_csv(input_dir, output_dir, data_type)


def main():
    root = tk.Tk()
    root.title("Training Data CSV Generator")
    root.geometry("400x200")

    # Create buttons
    negative_btn = tk.Button(
        root,
        text="Generate Negative Training Data",
        command=lambda: select_directories_and_generate("negative"),
        height=2,
        width=30
    )
    negative_btn.pack(pady=20)

    positive_btn = tk.Button(
        root,
        text="Generate Positive Training Data",
        command=lambda: select_directories_and_generate("positive"),
        height=2,
        width=30
    )
    positive_btn.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()