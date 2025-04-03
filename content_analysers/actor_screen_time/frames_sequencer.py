import os
import tkinter as tk
from tkinter import filedialog, messagebox


class FramesSequencer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

    def select_and_rename_frames(self):
        # Ask user to select a directory
        directory = filedialog.askdirectory(title="Select Directory with JPG Files")
        if not directory:
            return False

        # Get all JPG files in the directory
        jpg_files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
        if not jpg_files:
            messagebox.showwarning("No JPG Files", "No JPG files found in the selected directory.")
            return False

        # Sort files to ensure proper sequencing
        jpg_files.sort()

        # Rename files sequentially
        for index, filename in enumerate(jpg_files):
            old_path = os.path.join(directory, filename)
            new_name = f"frame{index}.jpg"
            new_path = os.path.join(directory, new_name)

            # Handle potential name conflicts
            if os.path.exists(new_path) and new_path != old_path:
                temp_name = f"temp_{index}.jpg"
                temp_path = os.path.join(directory, temp_name)
                os.rename(old_path, temp_path)
            else:
                os.rename(old_path, new_path)

        # Rename any temp files
        for index, filename in enumerate(jpg_files):
            temp_name = f"temp_{index}.jpg"
            temp_path = os.path.join(directory, temp_name)
            if os.path.exists(temp_path):
                new_name = f"frame{index}.jpg"
                new_path = os.path.join(directory, new_name)
                os.rename(temp_path, new_path)

        messagebox.showinfo("Success", f"Renamed {len(jpg_files)} files successfully.")
        return True


def main():
    sequencer = FramesSequencer()
    sequencer.select_and_rename_frames()


if __name__ == "__main__":
    main()