"""
Anagram Checker — class with the logic to validate and find anagrams.
"""

import os


class AnagramChecker:
    def __init__(self, word_list_path=None):
        if word_list_path is None:
            word_list_path = os.path.join(os.path.dirname(__file__), "words.txt")
        with open(word_list_path, "r", encoding="utf-8") as f:
            self.word_list = {line.strip().lower() for line in f if line.strip()}

    def is_valid_word(self, word):
        return word.strip().lower() in self.word_list

    def is_anagram(self, word1, word2):
        w1 = word1.strip().lower()
        w2 = word2.strip().lower()
        if w1 == w2:
            return False
        return sorted(w1) == sorted(w2)

    def get_anagrams(self, word):
        target = word.strip().lower()
        return [w for w in self.word_list if self.is_anagram(target, w)]
