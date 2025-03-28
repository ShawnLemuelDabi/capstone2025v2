import yt_dlp
import os
import re
from tkinter import Tk, filedialog


def parse_download_list(file_path):
    """Parses the downloadList.txt file."""
    try:
        with open(file_path, "r") as file:
            content = file.read()

        # Extract lists using regular expressions
        lists = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        result = []
        for lst in lists:
            urls = re.findall(r'"(.*?)"', lst)
            result.append(urls)
        return result

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return None


def download_tiktok_videos(file_path, output_base_dir, failed_file="failedDownload.txt"):
    """Downloads new TikTok videos with automatic format selection and video ID filename."""
    urls_data = parse_download_list(file_path)

    if urls_data is None:
        return

    # Create subdirectory based on the input filename
    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(output_base_dir, file_name_without_ext)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'progress_hooks': [
            lambda d: print(f"Downloading: {d['filename']}", end='\r') if d['status'] == 'downloading' else None],
        'download_archive': os.path.join(output_dir, 'downloaded_videos.txt'),
    }

    failed_downloads = []
    new_downloads = 0

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url_list in urls_data:
            for url in url_list:
                try:
                    info_dict = ydl.extract_info(url, download=False)
                    video_id = info_dict.get('id')
                    if video_id and os.path.exists(os.path.join(output_dir, f"{video_id}.*")):
                        print(f"Skipping already downloaded video: {url}")
                        continue  # Skip if a file with the video ID already exists

                    ydl.download([url])
                    print(f"\nDownloaded new video: {url}")
                    new_downloads += 1
                except yt_dlp.utils.DownloadError as e:
                    failed_downloads.append(url)
                    if "Requested format is not available" in str(e):
                        print(f"\nFormat not available for {url}. Trying fallback format.")
                        try:
                            ydl_opts['format'] = 'best'
                            ydl.download([url])
                            print(f"\nDownloaded new video (fallback format): {url}")
                            new_downloads += 1
                        except yt_dlp.utils.DownloadError as fallback_e:
                            print(f"\nFallback failed for {url}: {fallback_e}")
                    else:
                        print(f"\nError downloading {url}: {e}")
                except Exception as e:
                    failed_downloads.append(url)
                    print(f"\nAn unexpected error occurred: {e}")

    failed_file_path = os.path.join(output_dir, failed_file)
    if failed_downloads:
        with open(failed_file_path, "w") as f:
            for url in failed_downloads:
                f.write(f"{url}\n")
        print(f"\nFailed downloads written to {failed_file_path}")

    if new_downloads > 0:
        print(f"\nSuccessfully downloaded {new_downloads} new video(s).")
    else:
        print("\nNo new videos were downloaded.")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main tkinter window

    initial_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs"
    file_path = filedialog.askopenfilename(
        title="Select a download list file",
        initialdir=initial_dir,
        filetypes=[("Text files", "*.txt")]
    )

    if file_path:
        output_base_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\downloads"
        download_tiktok_videos(file_path, output_base_dir)
    else:
        print("No file selected.")