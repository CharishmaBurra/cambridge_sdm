"""
input_handler.py

Module responsible for reading and parsing JSON alignment files, and loading them into
Pandas DataFrames.
"""

import os
import glob
import json
import pandas as pd


class InputHandler:
    """
    A class that provides methods to load JSON alignment files and
    return them as a consolidated DataFrame.
    """

    @staticmethod
    def extract_phoneme_data(json_file: str):
        """
        Parses one JSON alignment file to extract phoneme-level records.
        If the JSON is invalid or doesn't decode to a dictionary, returns an empty list.
        """
        if not json_file.lower().endswith('.json'):
            print(f"SKIP (not .json): {json_file}")
            return []

        print(f"DEBUG: Processing file: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"WARNING: Could not parse {json_file} as valid JSON. Skipping. Error: {exc}")
            return []

        if not isinstance(data, dict):
            print(f"WARNING: JSON in {json_file} is {type(data)}, not dict. Skipping file.")
            return []

        if 'result' not in data:
            print(f"WARNING: No 'result' key in {json_file}. Skipping.")
            return []

        if not isinstance(data['result'], dict):
            print(f"WARNING: 'result' in {json_file} is {type(data['result'])}, expected dict. Skipping.")
            return []

        segments = data['result'].get('segments', [])
        records = []

        for segment in segments:
            words = segment.get('words', [])
            for word in words:
                phones = word.get('phones', [])
                for phone in phones:
                    records.append({
                        'word': word.get('word_normalized'),
                        'phone': phone.get('phone'),
                        'class': phone.get('class'),
                        'duration': phone.get('duration')
                    })

        print(f"DEBUG: Extracted {len(records)} records from {json_file}")
        return records

    @staticmethod
    def load_dataset(local_dir: str) -> pd.DataFrame:
        """
        Recursively reads all *.json files in local_dir (including subfolders)
        and combines phoneme records into a single DataFrame.
        """
        pattern = os.path.join(local_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)

        if not json_files:
            print(f"WARNING: No JSON files found under {local_dir}")

        all_records = []
        for jf in json_files:
            phoneme_records = InputHandler.extract_phoneme_data(jf)
            all_records.extend(phoneme_records)

        return pd.DataFrame(all_records)
