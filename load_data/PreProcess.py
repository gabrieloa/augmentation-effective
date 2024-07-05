import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

STOPWORDS_PATTERN = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
SPACE_PATTERN = re.compile(r'\s{2,}')
NUMBER_PATTERN = re.compile(r'\b\d+\b')


def process_regex(sentence: str) -> str:
    sentence_web = re.sub(r"(\\u2019|\\u2018)", "'", sentence)
    sentence_web = re.sub(r"(\\u002c)", " ", sentence_web)
    sentence_punctuation = PUNCTUATION_PATTERN.sub(' ', sentence_web.lower())
    sentence_space = SPACE_PATTERN.sub(' ', sentence_punctuation)
    sentence_number = NUMBER_PATTERN.sub('NUMBER', sentence_space)
    sentence_final = STOPWORDS_PATTERN.sub(' ', sentence_number)
    return sentence_final.lower()
