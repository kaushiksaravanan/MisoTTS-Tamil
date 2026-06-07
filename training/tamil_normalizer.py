"""
Tamil text normalization for TTS.

Handles Tamil-specific preprocessing:
  - Number to words (Tamil numerals)
  - Abbreviation expansion
  - Punctuation normalization
  - Mixed Tamil-English text handling
  - Unicode normalization (NFC)
"""

import re
import unicodedata


# Tamil digit words
TAMIL_ONES = ['', 'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து',
              'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது']
TAMIL_TENS = ['', 'பத்து', 'இருபது', 'முப்பது', 'நாற்பது', 'ஐம்பது',
              'அறுபது', 'எழுபது', 'எண்பது', 'தொண்ணூறு']
TAMIL_HUNDREDS = 'நூறு'
TAMIL_THOUSANDS = 'ஆயிரம்'
TAMIL_LAKHS = 'லட்சம்'
TAMIL_CRORES = 'கோடி'


def number_to_tamil_words(n: int) -> str:
    """Convert an integer to Tamil words."""
    if n == 0:
        return 'பூஜ்ஜியம்'
    if n < 0:
        return 'கழித்தல் ' + number_to_tamil_words(-n)

    parts = []

    if n >= 10000000:
        crores = n // 10000000
        parts.append(number_to_tamil_words(crores) + ' ' + TAMIL_CRORES)
        n %= 10000000

    if n >= 100000:
        lakhs = n // 100000
        parts.append(number_to_tamil_words(lakhs) + ' ' + TAMIL_LAKHS)
        n %= 100000

    if n >= 1000:
        thousands = n // 1000
        if thousands == 1:
            parts.append(TAMIL_THOUSANDS)
        else:
            parts.append(number_to_tamil_words(thousands) + ' ' + TAMIL_THOUSANDS)
        n %= 1000

    if n >= 100:
        hundreds = n // 100
        if hundreds == 1:
            parts.append(TAMIL_HUNDREDS)
        else:
            parts.append(TAMIL_ONES[hundreds] + ' ' + TAMIL_HUNDREDS)
        n %= 100

    if n >= 10:
        tens = n // 10
        ones = n % 10
        if ones == 0:
            parts.append(TAMIL_TENS[tens])
        else:
            parts.append(TAMIL_TENS[tens] + ' ' + TAMIL_ONES[ones])
    elif n > 0:
        parts.append(TAMIL_ONES[n])

    return ' '.join(parts)


# Common Tamil abbreviations
ABBREVIATIONS = {
    'திரு.': 'திருவாளர்',
    'திருமதி.': 'திருமதி',
    'செல்வி.': 'செல்வி',
    'டாக்டர்.': 'டாக்டர்',
    'பேராசிரியர்.': 'பேராசிரியர்',
    'ரூ.': 'ரூபாய்',
    'கி.மீ.': 'கிலோமீட்டர்',
    'செ.மீ.': 'சென்டிமீட்டர்',
    'கி.கி.': 'கிலோகிராம்',
}

# English letters to Tamil phonetic
ENGLISH_TO_TAMIL_PHONETIC = {
    'A': 'ஏ', 'B': 'பீ', 'C': 'சீ', 'D': 'டீ', 'E': 'ஈ',
    'F': 'எஃப்', 'G': 'ஜீ', 'H': 'எச்', 'I': 'ஐ', 'J': 'ஜே',
    'K': 'கே', 'L': 'எல்', 'M': 'எம்', 'N': 'என்', 'O': 'ஓ',
    'P': 'பீ', 'Q': 'கியூ', 'R': 'ஆர்', 'S': 'எஸ்', 'T': 'டீ',
    'U': 'யூ', 'V': 'வீ', 'W': 'டபிள்யூ', 'X': 'எக்ஸ்',
    'Y': 'வை', 'Z': 'ஜட்',
}


def normalize_tamil_text(text: str) -> str:
    """Full Tamil text normalization pipeline for TTS."""
    # Unicode normalization (NFC)
    text = unicodedata.normalize('NFC', text)

    # Expand abbreviations
    for abbr, expansion in ABBREVIATIONS.items():
        text = text.replace(abbr, expansion)

    # Convert numbers to Tamil words
    def replace_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            return number_to_tamil_words(num)
        except ValueError:
            return num_str

    text = re.sub(r'\d+', replace_number, text)

    # Handle English acronyms (all caps, 2+ letters) -> spell out
    def spell_acronym(match):
        letters = match.group(0)
        return ' '.join(ENGLISH_TO_TAMIL_PHONETIC.get(c, c) for c in letters)

    text = re.sub(r'\b[A-Z]{2,}\b', spell_acronym, text)

    # Normalize punctuation
    text = text.replace('—', ', ')
    text = text.replace('–', ', ')
    text = text.replace('…', '.')
    text = re.sub(r'["\"\"]', '', text)
    text = re.sub(r"['\'\']", '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_tamil_text(text: str) -> bool:
    """Check if text contains Tamil characters."""
    tamil_range = range(0x0B80, 0x0BFF + 1)
    for char in text:
        if ord(char) in tamil_range:
            return True
    return False


def segment_mixed_text(text: str) -> list:
    """
    Segment mixed Tamil-English text into chunks.
    Returns list of (text, language) tuples.
    """
    segments = []
    current = ""
    current_lang = None

    for char in text:
        if ord(char) in range(0x0B80, 0x0BFF + 1):
            lang = "ta"
        elif char.isalpha():
            lang = "en"
        else:
            if current:
                current += char
            continue

        if lang != current_lang and current:
            segments.append((current.strip(), current_lang))
            current = ""
        current += char
        current_lang = lang

    if current.strip():
        segments.append((current.strip(), current_lang))

    return segments
