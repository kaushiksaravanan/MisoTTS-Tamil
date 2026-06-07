"""
Tamil phoneme-aware tokenization helper.

The Llama 3.2 tokenizer was trained primarily on English/Latin text.
Tamil Unicode characters may be split into individual bytes, losing
phonological structure. This module provides:

1. Tamil phoneme segmentation (consonant + vowel sign = one unit)
2. Romanization fallback (ISO 15919) for better tokenizer coverage
3. Speaker-conditioned prompting for multi-speaker models

Key insight: MisoTTS uses [speaker_id] prefix + raw text. The tokenizer
handles Tamil characters via byte-fallback, which works but isn't optimal.
We add optional romanization mode that maps Tamil to ISO 15919 transliteration,
which the Llama tokenizer handles much better (since it was trained on romanized
Indian language text from the web).
"""

import re
from typing import Optional


# Tamil Unicode ranges
TAMIL_VOWELS = 'அஆஇஈஉஊஎஏஐஒஓஔ'
TAMIL_CONSONANTS = 'கஙசஞடணதநபமயரலவழளறன'
TAMIL_VOWEL_SIGNS = 'ாிீுூெேைொோௌ'
TAMIL_VIRAMA = '்'  # புள்ளி
TAMIL_ANUSVARA = 'ஂ'
TAMIL_VISARGA = 'ஃ'  # ஃ (aaytham)

# ISO 15919 Romanization mapping
VOWEL_MAP = {
    'அ': 'a', 'ஆ': 'ā', 'இ': 'i', 'ஈ': 'ī', 'உ': 'u', 'ஊ': 'ū',
    'எ': 'e', 'ஏ': 'ē', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'ō', 'ஔ': 'au',
}

CONSONANT_MAP = {
    'க': 'ka', 'ங': 'ṅa', 'ச': 'ca', 'ஞ': 'ña', 'ட': 'ṭa', 'ண': 'ṇa',
    'த': 'ta', 'ந': 'na', 'ப': 'pa', 'ம': 'ma', 'ய': 'ya', 'ர': 'ra',
    'ல': 'la', 'வ': 'va', 'ழ': 'ḻa', 'ள': 'ḷa', 'ற': 'ṟa', 'ன': 'ṉa',
}

VOWEL_SIGN_MAP = {
    'ா': 'ā', 'ி': 'i', 'ீ': 'ī', 'ு': 'u', 'ூ': 'ū',
    'ெ': 'e', 'ே': 'ē', 'ை': 'ai',
    'ொ': 'o', 'ோ': 'ō', 'ௌ': 'au',
}


def romanize_tamil(text: str) -> str:
    """
    Convert Tamil text to ISO 15919 romanization.
    This gives better tokenizer coverage since Llama was trained on
    romanized Indian text from the web (Hinglish, Tanglish, etc.)
    """
    result = []
    i = 0
    while i < len(text):
        char = text[i]

        if char in VOWEL_MAP:
            result.append(VOWEL_MAP[char])
            i += 1

        elif char in CONSONANT_MAP:
            base = CONSONANT_MAP[char][:-1]  # remove inherent 'a'

            if i + 1 < len(text):
                next_char = text[i + 1]
                if next_char == TAMIL_VIRAMA:
                    result.append(base)
                    i += 2
                elif next_char in VOWEL_SIGN_MAP:
                    result.append(base + VOWEL_SIGN_MAP[next_char])
                    i += 2
                else:
                    result.append(base + 'a')  # inherent vowel
                    i += 1
            else:
                result.append(base + 'a')
                i += 1

        elif char == TAMIL_VISARGA:
            result.append('ḵ')
            i += 1

        elif char == TAMIL_ANUSVARA:
            result.append('ṁ')
            i += 1

        elif char == TAMIL_VIRAMA:
            i += 1

        else:
            result.append(char)
            i += 1

    return ''.join(result)


def segment_tamil_syllables(text: str) -> list:
    """
    Segment Tamil text into syllable-like units.
    Each unit is: consonant + optional virama + optional vowel sign
    or standalone vowel.
    """
    syllables = []
    i = 0
    current = ""

    while i < len(text):
        char = text[i]

        if char in TAMIL_VOWELS:
            if current:
                syllables.append(current)
            current = char
            i += 1

        elif char in TAMIL_CONSONANTS:
            if current:
                syllables.append(current)
            current = char
            i += 1

            # Consume virama + next consonant (conjuncts)
            while i < len(text) and text[i] == TAMIL_VIRAMA:
                current += text[i]
                i += 1
                if i < len(text) and text[i] in TAMIL_CONSONANTS:
                    current += text[i]
                    i += 1

            # Consume vowel sign
            if i < len(text) and text[i] in TAMIL_VOWEL_SIGNS:
                current += text[i]
                i += 1

        elif char in (TAMIL_VIRAMA, TAMIL_ANUSVARA, TAMIL_VISARGA):
            current += char
            i += 1

        elif char in TAMIL_VOWEL_SIGNS:
            current += char
            i += 1

        else:
            if current:
                syllables.append(current)
                current = ""
            syllables.append(char)
            i += 1

    if current:
        syllables.append(current)

    return syllables


def format_text_for_tts(
    text: str,
    speaker: int,
    romanize: bool = False,
) -> str:
    """
    Format text for MisoTTS input.
    Optionally romanizes Tamil for better tokenizer coverage.
    """
    if romanize:
        # Split into Tamil and non-Tamil segments
        parts = re.split(r'([^஀-௿]+)', text)
        processed = []
        for part in parts:
            if any('஀' <= c <= '௿' for c in part):
                processed.append(romanize_tamil(part))
            else:
                processed.append(part)
        text = ''.join(processed)

    return f"[{speaker}] {text.lstrip()}"
