import time
import keyboard
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException


def extract_pinned_status(card):
    """Extract whether video is pinned"""
    try:
        return "Yes" if card.find_element(By.XPATH, './/span[@data-icon="Pin"]') else "No"
    except NoSuchElementException:
        return "No"


def extract_caption(card):
    """Extract video caption text"""
    try:
        caption = card.find_element(By.XPATH, './/a[contains(@href, "/video/")]').text
        return caption[:200] + '...' if len(caption) > 200 else caption
    except NoSuchElementException:
        return "N/A"


def extract_privacy(card):
    """Extract privacy status (Public/Private)"""
    try:
        return card.find_element(
            By.XPATH,
            './/span[@data-icon="Earth" or @data-icon="Lock"]/following-sibling::div[contains(@class, "TUXButton-label")]'
        ).text
    except NoSuchElementException:
        return "N/A"


def extract_metrics(card):
    """Extract views, likes, comments"""
    metrics = {'views': '0', 'likes': '0', 'comments': '0'}
    try:
        elements = card.find_elements(
            By.XPATH,
            './/div[contains(@data-tt, "components_ItemRow_FlexCenter")]//span[contains(@class, "TUXText")]'
        )
        metrics['views'] = elements[0].text.replace(',', '') if len(elements) > 0 else '0'
        metrics['likes'] = elements[1].text.replace(',', '') if len(elements) > 1 else '0'
        metrics['comments'] = elements[2].text.replace(',', '') if len(elements) > 2 else '0'
    except Exception:
        pass
    return metrics


def auto_scroll_and_extract(driver):
    container_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[2]/div/div/div[2]'
    all_videos_data = []
    processed_video_ids = set()
    scroll_pause_time = 3
    max_scrolls_without_new_data = 10
    scrolls_without_new_data = 0
    should_continue = True

    print("Press ENTER at any time to stop scrolling and save current results...")

    try:
        container = WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.XPATH, container_xpath)))
    except TimeoutException:
        print(f"Container not found: {container_xpath}")
        return []

    while should_continue and scrolls_without_new_data < max_scrolls_without_new_data:
        if keyboard.is_pressed('enter'):
            print("\nEarly termination requested...")
            should_continue = False
            break

        video_cards = driver.find_elements(
            By.XPATH,
            '//div[contains(@class, "css-153feq8") and contains(@data-tt, "components_RowLayout_FlexRow")]'
        )
        newly_processed_count = 0

        for card in video_cards:
            try:
                video_link = card.find_element(By.XPATH, './/a[contains(@href, "/video/")]')
                video_url = video_link.get_attribute("href")
                video_id = video_url.split("/video/")[1].split("?")[0] if video_url else None

                if not video_id or video_id in processed_video_ids:
                    continue

                video_data = {
                    'video_id': video_id,
                    'video_url': video_url,
                    'date_time': extract_datetime(card),
                    'pinned': extract_pinned_status(card),
                    'views': extract_metrics(card)['views'],
                    'likes': extract_metrics(card)['likes'],
                    'comments': extract_metrics(card)['comments'],
                    'caption': extract_caption(card),
                    'privacy': extract_privacy(card)
                }

                all_videos_data.append(video_data)
                processed_video_ids.add(video_id)
                newly_processed_count += 1
                print(f"✅ Scraped {video_id}")

            except Exception as e:
                print(f"⚠️ Card processing error: {str(e)}")
                continue

        if newly_processed_count > 0:
            scrolls_without_new_data = 0
        else:
            scrolls_without_new_data += 1
            print(f"No new videos ({scrolls_without_new_data}/{max_scrolls_without_new_data})")

        driver.execute_script("arguments[0].scrollTop += 500;", container)
        time.sleep(scroll_pause_time)

    print(f"\nScraping complete. Collected {len(all_videos_data)} unique videos.")
    return all_videos_data


def extract_datetime(card):
    """Extract upload date/time"""
    try:
        return card.find_element(
            By.XPATH,
            './/span[contains(@class, "TUXText") and contains(@data-tt, "components_PublishStageLabel_TUXText")]'
        ).text
    except NoSuchElementException:
        return "N/A"