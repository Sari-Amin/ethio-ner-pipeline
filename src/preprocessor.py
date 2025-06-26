# preprocessor.py

import re
import unicodedata
import json
import pandas as pd
import os
from typing import List, Dict


class MessagePreprocessor:
    def __init__(self):
        pass

    def load_raw_data(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"http\S+|www\S+", "", text)                        # URLs
        text = re.sub(r"[^\w፡።፣፤፥፦መምሚማሞወዉዊዋዎተቱቲታቶትነኑኒናኖንኘኙኚኛኞኝከኩኪካኮክኸኹኺኻኾኽየዩዪያዮይደዱዲዳዶድፈፉፊፋፎፍፐፑፒፓፖፕ\s]", " ", text)
        text = re.sub(r"[።፣፤፥፦]+", ".", text)                        # Amharic punctuation to dot
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        cleaned = []
        seen = set()

        for msg in messages:
            raw = msg.get("Message", "")
            clean = self.normalize_text(raw)

            if clean and clean not in seen and len(clean.split()) >= 3:
                msg["clean_text"] = clean
                cleaned.append(msg)
                seen.add(clean)

        return cleaned

    def save_to_csv(self, data: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(data)
        columns = ["ChannelTitle", "ChannelUsername", "ID", "Date", "views", "Message", "clean_text", "MediaPath"]
        df = df[[col for col in columns if col in df.columns]]
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(data)}cleaned CSV to {output_path}")

    def save_to_json(self, data: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved cleaned JSON to {output_path}")
