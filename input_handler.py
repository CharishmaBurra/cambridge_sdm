"""
Complete input handling with both base and enhanced functionality.
"""

import os
import glob
import json
import pandas as pd
from typing import List, Dict, Any

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class InputHandler:
    """
    Base input handler class with core functionality
    """
    
    @staticmethod
    def extract_phoneme_data(json_file: str) -> List[Dict[str, Any]]:
        """Base phoneme data extraction"""
        if not json_file.lower().endswith('.json'):
            print(f"SKIP (not .json): {json_file}")
            return []

        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if isinstance(data, list):
                print(f"WARNING: {json_file} contains a JSON array instead of object. Skipping.")
                return []
            
            if not isinstance(data, dict) or 'result' not in data:
                print(f"WARNING: {json_file} missing 'result' key or not a dictionary. Skipping.")
                return []
            
            if not isinstance(data['result'], dict):
                print(f"WARNING: 'result' in {json_file} is not a dictionary. Skipping.")
                return []

            segments = data['result'].get('segments', [])
            records = []

            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                
                words = segment.get('words', [])
                for word_idx, word in enumerate(words):
                    if not isinstance(word, dict):
                        continue
                    
                    word_text = word.get('word_normalized', '')
                    phones = word.get('phones', [])
                    word_length = len(phones)
                
                    for phone_idx, phone in enumerate(phones):
                        if not isinstance(phone, dict):
                            continue
                        
                        phone_data = phone.get('phone')
                        duration = phone.get('duration')
                    
                        if phone_data is None or duration is None:
                            continue
                        
                        records.append({
                            'file': os.path.basename(json_file),
                            'word': word_text,
                            'phone': phone_data,
                            'class': phone.get('class'),
                            'duration': duration,
                            'position_in_word': phone_idx,
                            'word_length': word_length,
                            'is_first_phone': phone_idx == 0,
                            'is_last_phone': phone_idx == word_length - 1,
                            'word_position': word_idx
                        })

            return records

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"WARNING: Could not parse {json_file}. Error: {exc}")
            return []
        except Exception as exc:
            print(f"ERROR: Unexpected error processing {json_file}. Error: {exc}")
            return []

    @staticmethod
    def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
        """Base contextual features"""
        vowels = {'a', 'e', 'i', 'o', 'u'}
        df['is_vowel'] = df['phone'].str.lower().isin(vowels).astype(int)
        df['is_silence'] = (df['phone'] == 'SIL').astype(int)
        df['is_consonant'] = (df['class'] == 'C').astype(int)
        df['position_ratio'] = df['position_in_word'] / df['word_length']
        
        df['prev_phone'] = df.groupby(['file', 'word'])['phone'].shift(1)
        df['next_phone'] = df.groupby(['file', 'word'])['phone'].shift(-1)
        df['prev_is_vowel'] = df['prev_phone'].str.lower().isin(vowels).fillna(0).astype(int)
        df['next_is_vowel'] = df['next_phone'].str.lower().isin(vowels).fillna(0).astype(int)
        
        return df

    @classmethod
    def load_dataset(cls, local_dir: str) -> pd.DataFrame:
        """Base dataset loading"""
        try:
            pattern = os.path.join(local_dir, '**', '*.json')
            json_files = glob.glob(pattern, recursive=True)
            
            if not json_files:
                print(f"WARNING: No JSON files found under {local_dir}")
                return pd.DataFrame()

            all_records = []
            valid_files = 0
            
            for jf in json_files:
                records = cls.extract_phoneme_data(jf)
                if records:
                    valid_files += 1
                    all_records.extend(records)
                    
            print(f"Processed {valid_files}/{len(json_files)} valid JSON files")
            
            if not all_records:
                return pd.DataFrame()
                
            df = pd.DataFrame(all_records)
            return cls.add_contextual_features(df)
            
        except Exception as exc:
            print(f"ERROR: Failed to load dataset from {local_dir}. Error: {exc}")
            return pd.DataFrame()


class EnhancedInputHandler(InputHandler):
    """Enhanced input handler with additional features"""
    
    @staticmethod
    def extract_phoneme_data(json_file: str) -> List[Dict[str, Any]]:
        """Enhanced phoneme data extraction with additional features"""
        records = InputHandler.extract_phoneme_data(json_file)
        if not records:
            return records
            
        # Add phonological features
        for record in records:
            phone = record['phone'].lower()
            
            # Basic phonological features
            record['is_voiced'] = int(phone in ['b', 'd', 'g', 'v', 'ð', 'z', 'ʒ', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j'])
            record['is_plosive'] = int(phone in ['p', 'b', 't', 'd', 'k', 'g'])
            record['is_fricative'] = int(phone in ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h'])
            record['is_nasal'] = int(phone in ['m', 'n', 'ŋ'])
            
            # Approximate stress (1 for vowels, 0 for consonants)
            record['has_stress'] = int(phone in ['a', 'e', 'i', 'o', 'u'])
            
        return records

    @staticmethod
    def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced contextual features"""
        df = InputHandler.add_contextual_features(df)
        
        # Phone bigrams
        df['phone_bigram'] = df['phone'] + '_' + df['next_phone'].fillna('END')
        
        # Duration ratios
        df['duration_ratio'] = df['duration'] / df.groupby('file')['duration'].transform('mean')
        
        # Positional features
        df['is_word_initial'] = (df['position_in_word'] == 0).astype(int)
        df['is_word_final'] = (df['position_in_word'] == df['word_length'] - 1).astype(int)
        
        # Utterance position features
        df['position_in_utterance'] = df.groupby('file').cumcount()
        df['utterance_length'] = df.groupby('file')['phone'].transform('count')
        df['position_ratio_utterance'] = df['position_in_utterance'] / df['utterance_length']
        
        return df

    @classmethod
    def load_dataset(cls, local_dir: str) -> pd.DataFrame:
        """Enhanced dataset loading"""
        try:
            pattern = os.path.join(local_dir, '**', '*.json')
            json_files = glob.glob(pattern, recursive=True)
            
            if not json_files:
                print(f"WARNING: No JSON files found under {local_dir}")
                return pd.DataFrame()

            all_records = []
            valid_files = 0
            
            for jf in json_files:
                records = cls.extract_phoneme_data(jf)
                if records:
                    valid_files += 1
                    all_records.extend(records)
                    
            print(f"Processed {valid_files}/{len(json_files)} valid JSON files")
            
            if not all_records:
                return pd.DataFrame()
                
            df = pd.DataFrame(all_records)
            return cls.add_contextual_features(df)
            
        except Exception as exc:
            print(f"ERROR: Failed to load dataset from {local_dir}. Error: {exc}")
            return pd.DataFrame()