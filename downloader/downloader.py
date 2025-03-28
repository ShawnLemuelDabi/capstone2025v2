import yt_dlp
import os
import re
from tkinter import Tk, filedialog
from tqdm import tqdm  # Import the tqdm library for progress bars


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
        print(f"❌ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"⚠️ An unexpected error occurred during parsing: {e}")
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

    downloaded_videos_file = os.path.join(output_dir, 'downloaded_videos.txt')
    downloaded_ids = set()
    if os.path.exists(downloaded_videos_file):
        with open(downloaded_videos_file, 'r') as f:
            for line in f:
                match = re.search(r'^(.*?)\s', line)  # Extract video ID from the start
                if match:
                    downloaded_ids.add(match.group(1))

    all_urls_to_download = [url for url_list in urls_data for url in url_list]
    total_videos = len(all_urls_to_download)
    new_downloads_count = 0
    failed_downloads = []

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'progress_hooks': [],  # Initialize empty, will be extended
        'download_archive': downloaded_videos_file,
    }

    def progress_hook(d):
        nonlocal current_video_name
        if d['status'] == 'downloading':
            filename = d.get('filename')
            if filename:
                current_video_name = os.path.basename(filename)
        elif d['status'] == 'finished':
            print(f"\n✅ Finished downloading: {current_video_name}")
        elif d['status'] == 'error':
            print(f"\n❌ Error downloading: {d.get('filename', 'unknown video')}")

    ydl_opts['progress_hooks'].append(progress_hook)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        pbar = tqdm(total=total_videos, initial=len(downloaded_ids), unit="video", desc="Downloading Videos")
        video_index = 0
        for url_list in urls_data:
            for url in url_list:
                video_index += 1
                try:
                    info_dict = ydl.extract_info(url, download=False)
                    video_id = info_dict.get('id')
                    video_title = info_dict.get('title', 'Unknown Title')
                    current_video_name = f"{video_title} ({video_id})"

                    if video_id in downloaded_ids and os.path.exists(os.path.join(output_dir, f"{video_id}.*")):
                        pbar.update(1)
                        pbar.set_postfix(current=current_video_name, downloaded=new_downloads_count, failed=len(failed_downloads), remaining=total_videos - len(downloaded_ids) - new_downloads_count - len(failed_downloads))
                        continue  # Skip if already downloaded

                    pbar.set_postfix(current=current_video_name, downloaded=new_downloads_count, failed=len(failed_downloads), remaining=total_videos - len(downloaded_ids) - new_downloads_count - len(failed_downloads))
                    ydl.download([url])
                    new_downloads_count += 1
                    downloaded_ids.add(video_id) # Add to the set after successful download
                    pbar.update(1)

                except yt_dlp.utils.DownloadError as e:
                    failed_downloads.append(url)
                    pbar.update(1)
                    pbar.set_postfix(current="Failed: " + url, downloaded=new_downloads_count, failed=len(failed_downloads), remaining=total_videos - len(downloaded_ids) - new_downloads_count - len(failed_downloads))
                    if "Requested format is not available" in str(e):
                        print(f"\n⚠️ Format not available for {url}. Trying fallback format.")
                        try:
                            ydl_opts['format'] = 'best'
                            ydl.download([url])
                            print(f"\n✅ Downloaded new video (fallback format): {url}")
                            new_downloads_count += 1
                            downloaded_ids.add(info_dict.get('id')) # Add after fallback success
                            pbar.update(1)
                        except yt_dlp.utils.DownloadError as fallback_e:
                            print(f"\n❌ Fallback failed for {url}: {fallback_e}")
                    else:
                        print(f"\n❌ Error downloading {url}: {e}")
                except Exception as e:
                    failed_downloads.append(url)
                    pbar.update(1)
                    pbar.set_postfix(current="Error: " + url, downloaded=new_downloads_count, failed=len(failed_downloads), remaining=total_videos - len(downloaded_ids) - new_downloads_count - len(failed_downloads))
                    print(f"\n⚠️ An unexpected error occurred with {url}: {e}")

        pbar.close()

    failed_file_path = os.path.join(output_dir, failed_file)
    if failed_downloads:
        with open(failed_file_path, "w") as f:
            for url in failed_downloads:
                f.write(f"{url}\n")
        print(f"\nℹ️ Failed downloads recorded in: {failed_file_path}")

    print(f"\n🎬 Summary:")
    print(f"  ✅ Successfully downloaded {new_downloads_count} new video(s).")
    if failed_downloads:
        print(f"  ❌ Failed to download {len(failed_downloads)} video(s). Check {failed_file_path} for details.")
    else:
        print("  🎉 No downloads failed.")


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
        print("🚫 No file selected. Exiting.")