import time
from telegram_scraper import TelegramScraper

if __name__ == "__main__":
    scraper = TelegramScraper(config_path="config.yaml")
    scraper.connect()

    for channel in scraper.channels:
        msgs = scraper.fetch_messages(channel)
        scraper.save_raw_messages(channel, msgs)
        time.sleep(1.5)

    scraper.close()
