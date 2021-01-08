import json
import re
from nltk import pos_tag
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

all_words = set(words.words())
stop_words = set(stopwords.words('english'))
other_words = ['nothing', 'learn', 'online', 'course', 'thank']

# Remove digits and characters
def cleanText(text):
  text = text.replace("\n", " ")
  text = text.replace("|", "")
  text = re.sub("(\\d|\\W)+", " ", text)
  text = text.lower()
  text = text.strip()
  return text

# Remove stop-words and non-English words
def filterWords(text):
  word_tokens = word_tokenize(text)
  filtered = []
  for w in word_tokens:
    if w not in stop_words and w in all_words and w not in other_words:
      filtered.append(w)
  return " ".join(filtered)

# Get nouns only
def getNouns(text):
  word_tokens = word_tokenize(text)
  adj_only = pos_tag(word_tokens)
  filtered = []
  for word, tag in adj_only:
    if (tag == 'NN'):
      filtered.append(word)
  return " ".join(filtered)

# Start
with open('texts.json') as input_file:
  file = input_file.read()

array = json.loads(file)
results = []

for arr in array['results']:
  texts = []
  for text in arr['reviewTexts']:
    text = getNouns(filterWords(cleanText(text)))
    if text != "" and text not in texts:
      texts.append(text)
  results.append({
    "course_id": arr['course_id'],
    "reviewTexts": texts
  })

with open('data_fixed/output_cleaned.json', 'w') as output_file:
  output_file.write(json.dumps(results, indent=2))