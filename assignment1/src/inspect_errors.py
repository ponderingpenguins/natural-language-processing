import json

# load the jsonl file
jsonl_svm = "../output/misclassified_linear_svm.jsonl"
jsonl_lr = "../output/misclassified_logistic_regression.jsonl"


def load_jsonl(file_path: str) -> list[dict[str, object]]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


misclassified_svm = load_jsonl(jsonl_svm)
misclassified_lr = load_jsonl(jsonl_lr)

# print("Misclassified samples for Linear SVM:")
# for sample in misclassified_svm[:5]:  # print first 5 samples
#     print(sample)

# {'index': 15, 'true_label': 4, 'true_class': 'Sci/Tech', 'predicted_label': 3, 'predicted_class': 'Business', 'raw_text': "Teenage T. rex's monster growth Tyrannosaurus rex achieved its massive size due to an enormous growth spurt during its adolescent years.", 'preprocessed_text': 'teenage rex monster growth tyrannosaurus rex achieved massive size due enormous growth spurt adolescent year'}

# print samples that have predicted class as business and true class as sci/tech for logistic regression
max_n = 10
count = 0
print("Business predicted as Sci/Tech:")
for sample in misclassified_lr:  # print first 5 samples
    if sample["predicted_class"] == "Business" and sample["true_class"] == "Sci/Tech":
        print("preprocessed_text: " + sample["preprocessed_text"] + "\n")
        print("raw_text: " + sample["raw_text"] + "\n")
        count += 1
        if count >= max_n:
            break

print("Sci/Tech predicted as Business:")
count = 0
for sample in misclassified_lr:  # print first 5 samples
    if sample["predicted_class"] == "Sci/Tech" and sample["true_class"] == "Business":
        print("preprocessed_text: " + sample["preprocessed_text"] + "\n")
        print("raw_text: " + sample["raw_text"] + "\n")
        count += 1
        if count >= max_n:
            break
