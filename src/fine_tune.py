#%%
import datasets

finer_train = datasets.load_dataset("nlpaueb/finer-139", split="train")
finer_tag_names=finer_train.features['ner_tags'].feature.names
print(finer_tag_names[:25])

#%%
