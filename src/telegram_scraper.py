from datetime import datetime
import json
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from typing import List, Dict
import os
import yaml


class TelegramScraper:
    def __init__(self, config_path: str):
        """
        Initializes the scraper using config.yaml
        """
        self.config = self._load_config(config_path)
        self.api_id = self.config['telegram']['api_id']
        self.api_hash = self.config['telegram']['api_hash']
        self.session_name = self.config['telegram'].get('session_name', 'ethio_ner')
        self.channels = self.config['telegram']['channels']
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)

    def _load_config(self, path: str) -> Dict:
        """
        Load YAML config from the given path.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def connect(self):
        """
        Connect to Telegram client.
        """
        self.client.start()

    def fetch_messages(self, channel: str, limit: int = 10000) -> List[Dict]:
        """
        Fetch messages from a given Telegram channel.
        """
        messages_data = []

        for message in self.client.iter_messages(channel, limit=limit):
            if not message.message:
                continue
            media_path = None
            if message.media and isinstance(message.media, MessageMediaPhoto):
                media_path = f"data/media/{channel}_{message.id}.jpg"
                self.client.download_media(message, file=media_path)

            msg_dict = {
                "ChannelTitle": channel,
                "ChannelUsername": channel,
                "ID": message.id,
                "views": message.views,
                "Message": message.message,
                "Date": message.date.isoformat(),
                "MediaPath": media_path
            }
            messages_data.append(msg_dict)

        print(f"Fetched {len(messages_data)} messages from {channel}.")
        return messages_data

    def save_raw_messages(self, channel: str, messages: List[Dict]):
        """
        Save raw message data to data/raw/{channel}.json
        """
        filename = f"data/raw/{channel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        print(f"Saved messages to {filename}")

    def close(self):
        """
        Disconnect from the Telegram client.
        """
        self.client.disconnect()
