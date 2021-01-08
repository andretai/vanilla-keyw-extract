import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy

with open('data_fixed/output_cleaned.json') as input_file:
  inputFile = input_file.read()

converted = json.loads(inputFile)

final = []

for conv in converted:
  # collect vocabulary and word counts for idf
  print(conv)
  if conv['reviewTexts'] == []:
    continue
  cv = CountVectorizer()
  word_count_vector = cv.fit_transform(conv['reviewTexts'])
  # print(cv.vocabulary_.keys())

  # compute idf
  tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
  tfidf_transformer.fit(word_count_vector)
  # visualize idf
  df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf"])
  print(df_idf.sort_values(by=['idf'], ascending=False)[:5])

  array = []

  # tfidf
  docs = conv['reviewTexts']
  feature_names = cv.get_feature_names()
  for doc in docs:
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
    df = pd.DataFrame(tf_idf_vector.T.todense(), index=feature_names, columns=['tfidf'])
    df = df[(df.tfidf > .5)]
    # arr = df.sort_values(by=['tfidf'], ascending=False)
    for index, row in df.iterrows():
      array.append({
        "keyword": index,
        "tfidf": row['tfidf']
      })

  def sort_func(obj):
    return obj['tfidf']

  array.sort(reverse=True, key=sort_func)

  new_arr = []

  for ar in array:
    new_arr.append(ar['keyword'])

  res = [] 
  for i in new_arr: 
      if i not in res: 
          res.append(i) 
  final.append({
    "course_id": conv['course_id'],
    "keywords": res
  })

with open('results/keywords_weights.json', 'w') as output_file:
  output_file.write(json.dumps(final, indent=2))