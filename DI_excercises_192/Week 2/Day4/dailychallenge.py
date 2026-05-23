"""
Week 2 - Day 4 - Daily Challenge
Text and TextModification classes for text analysis.
"""

import os
import re
import string


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "she", "that", "the", "their", "them", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "will", "with", "you", "your", "do", "does", "did", "not",
    "no", "yes", "if", "so", "than", "then", "there", "these", "those",
    "am", "been", "being", "had", "having", "would", "could", "should",
    "shall", "may", "might", "must", "can", "about", "into", "over",
    "under", "again", "further", "once", "here", "how", "why",
}


class Text:
    def __init__(self, text):
        self.text = text

    def _words(self):
        return self.text.split()

    def word_frequency(self, word):
        words = self._words()
        count = words.count(word)
        if count == 0:
            return f"'{word}' no se encuentra en el texto."
        return count

    def most_common_word(self):
        words = self._words()
        if not words:
            return None
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return max(freq, key=freq.get)

    def unique_words(self):
        return list(set(self._words()))

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls(content)


class TextModification(Text):
    def remove_punctuation(self):
        translator = str.maketrans("", "", string.punctuation)
        return self.text.translate(translator)

    def remove_stop_words(self):
        words = self.text.split()
        filtered = [w for w in words if w.lower() not in STOP_WORDS]
        return " ".join(filtered)

    def remove_special_characters(self):
        return re.sub(r"[^A-Za-z0-9\s]", "", self.text)


if __name__ == "__main__":
    sample = "the quick brown fox jumps over the lazy dog the fox is quick"
    t = Text(sample)

    print(f"Texto: {sample}\n")
    print(f"Frecuencia de 'the': {t.word_frequency('the')}")
    print(f"Frecuencia de 'cat': {t.word_frequency('cat')}")
    print(f"Palabra más común: {t.most_common_word()}")
    print(f"Palabras únicas: {sorted(t.unique_words())}")

    words_path = os.path.join(os.path.dirname(__file__), "words.txt")
    if os.path.exists(words_path):
        t_file = Text.from_file(words_path)
        print(f"\nDesde archivo {os.path.basename(words_path)}:")
        print(f"  Palabra más común: {t_file.most_common_word()}")
        print(f"  Total de palabras únicas: {len(t_file.unique_words())}")

    print("\n--- TextModification ---")
    dirty = "Hello, world! The quick brown fox... jumps over the lazy dog. #python @2026"
    tm = TextModification(dirty)
    print(f"Original:                {dirty}")
    print(f"Sin puntuación:          {tm.remove_punctuation()}")
    print(f"Sin stop words:          {tm.remove_stop_words()}")
    print(f"Sin caracteres especiales: {tm.remove_special_characters()}")
