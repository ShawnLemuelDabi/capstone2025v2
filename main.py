from scraper.driver import initialize_driver
from scraper.login import manual_login
from scraper.scraper import auto_scroll_and_extract
from scraper.utils import save_results


def main():
    # Ask for account name
    account_name = input("Enter TikTok account name (leave blank to auto-detect): ").strip()

    driver = initialize_driver()
    try:
        # Login (will auto-detect if no name provided)
        detected_name = manual_login(driver)
        final_account_name = account_name or detected_name or "unknown_account"

        print(f"\nStarting scraper for account: {final_account_name}")
        input("Press Enter when ready to begin scraping...")

        # Run scraper
        data = auto_scroll_and_extract(driver)

        # Save results
        if data:
            csv_path = save_results(data, final_account_name)
            print(f"\nDone! Saved {len(data)} videos to:\n{csv_path}")
        else:
            print("\nNo videos were scraped.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'driver' in locals():
            driver.save_screenshot(f"outputs/error_{final_account_name}.png")
    finally:
        if 'driver' in locals():
            driver.quit()


if __name__ == "__main__":
    main()