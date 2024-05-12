from sentence_transformers import SentenceTransformer, util
import pandas as pd

bools = ["true", "false"]
word_list = ["apple", "banana", "orange", "true", "false"]
model = SentenceTransformer("all-MiniLM-L6-v2")

boolean_embeddings = model.encode(bools, convert_to_tensor=True)
word_embeddings = model.encode(word_list, convert_to_tensor=True)
cosine_similarities = util.cos_sim(word_embeddings, boolean_embeddings)

word_bool_df = pd.DataFrame(columns=["Word", "True_Score", "False_Score"])

for i, word in enumerate(word_list):
    true_score = cosine_similarities[i][0].item()
    false_score = cosine_similarities[i][1].item()
    word_bool_df = word_bool_df._append({"Word": word, "True_Score": true_score, "False_Score": false_score}, ignore_index=True)

print(word_bool_df)