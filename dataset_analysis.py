from datasets import load_dataset

# Loading dataset
dataset = load_dataset("ag_news")

# lenth of dataset
train_texts = dataset["train"]["text"]
lengths = [len(text.split()) for text in train_texts]

# calculate some statistical feature
mean_length = sum(lengths) / len(lengths)
median_length = sorted(lengths)[len(lengths) // 2]
percentile_90th = sorted(lengths)[int(len(lengths) * 0.9)]
percentile_995th = sorted(lengths)[int(len(lengths) * 0.995)]
percentile_999th = sorted(lengths)[int(len(lengths) * 0.999)]

print(f"Average Length: {mean_length}")
print(f"Median Length: {median_length}")
print(f"90% Length: {percentile_90th}")
print(f"99.5% Length: {percentile_995th}")
print(f"99.9% Length: {percentile_999th}")
