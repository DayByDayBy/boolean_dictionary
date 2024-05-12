import pandas as pd
from sentence_transformers import SentenceTransformer, util

bools = ["true", "false"]
word_list = ["apple", "banana", "orange"]

model = SentenceTransformer("all-MiniLM-L6-v2")

boolean_embeddings = model.encode(bools, convert_to_tensor=True)
word_embeddings = model.encode(word_list, convert_to_tensor=True)

cosine_similarities = util.cos_sim(word_embeddings, boolean_embeddings)

word_bool_list = []
for word, cosine_scores in zip(word_list, cosine_similarities):
    true_score = cosine_scores[0]
    false_score = cosine_scores[1]
    label = "true" if true_score > false_score else "false"
    word_bool_list.append([word, label, true_score, false_score])

word_bool_df = pd.DataFrame(word_bool_list, columns=['Word', 'Label', 'True_Score', 'False_Score'])

print(word_bool_df)

word_bool_df.to_csv('bool_dict_three_word_test.csv', index=False)








# from sentence_transformers import SentenceTransformer, util
# import pandas as pd

# bools = ["true", "false"]
# word_list = ["apple", "banana", "orange", "true", "false"]
# model = SentenceTransformer("all-MiniLM-L6-v2")

# boolean_embeddings = model.encode(bools, convert_to_tensor=True)
# word_embeddings = model.encode(word_list, convert_to_tensor=True)
# cosine_similarities = util.cos_sim(word_embeddings, boolean_embeddings)

# word_bool_df = pd.DataFrame(columns=["Word", "True_Score", "False_Score"])

# for i, word in enumerate(word_list):
#     true_score = cosine_similarities[i][0].item()
#     false_score = cosine_similarities[i][1].item()
#     word_bool_df = word_bool_df._append({"Word": word, "True_Score": true_score, "False_Score": false_score}, ignore_index=True)

# print(word_bool_df)