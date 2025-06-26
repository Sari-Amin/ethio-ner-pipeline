from telegram_scraper import TelegramScraper
from preprocessor import MessagePreprocessor
from datetime import datetime
import os

scraper = TelegramScraper(config_path="config.yaml")
preprocessor = MessagePreprocessor()

scraper.connect()

for channel in scraper.channels:
    messages = scraper.fetch_messages(channel, limit=10000)
    scraper.save_messages(channel, messages)

    # Clean
    cleaned = preprocessor.preprocess_messages(messages)

    # Save
    preprocessor.save_to_csv(cleaned, f"data/processed/{channel}_cleaned.csv")

scraper.close()
