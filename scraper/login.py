from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def manual_login(driver):
    print("Please login to TikTok Studio manually...")
    driver.get("https://www.tiktok.com/tiktokstudio/content")

    # Wait for login and detect account name
    WebDriverWait(driver, 120).until(
        EC.presence_of_element_located((By.XPATH, '//*[contains(text(), "Content")]'))
    )

    try:
        account_element = driver.find_element(
            By.XPATH, '//div[contains(@class, "account-name")]'
        )
        account_name = account_element.text.strip()
    except NoSuchElementException:
        account_name = None

    print("Login detected" + (f" (Account: {account_name})" if account_name else ""))
    return account_name  # Return the detected account name