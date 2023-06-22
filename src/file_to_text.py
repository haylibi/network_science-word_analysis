import contractions
import re
from textblob import TextBlob

def read_file(file_dir, encoding='utf-8', output='TextBlob'):
    # Fellowship Of The Ring for random network definition purposes:
    with open(file_dir, 'r', encoding=encoding) as file:
        # Read the content of the file
        # Fix the contractions (ex: They're -> They are / Bilbo's -> Bilbos)
        # Make it a textblob
        text = TextBlob(contractions.fix(file.read()))

    # setting every word to lowercase
    text.words = text.words.lower()

    # removing (".", "_") and ' when it starts a word
    for i in range(len(text.words)):
        word = text.words[i]
        word = word.replace(".", "").replace("_", "")
        text.words[i] = word

    return text if output=='TextBlob' else ' '.join(text.words)