import clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import json
from collections import defaultdict, Counter
from prompts_india import class_name_promt_map

def create_summarised_html(results, image_dir, outputdir="outputs"):

    class_to_images = defaultdict(list)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    precision_list = []
    recall_list = []
    f1_score_list = []
    summarised_results_path = os.path.join(outputdir, "clip_summarised_results.html")

    for image_name, info in results.items():
        class_to_images[info["ground_truth_class"]].append((image_name, info))
        confusion_matrix[info["ground_truth_class"]][info["predicted_class"]] += 1

    with open(summarised_results_path, "w") as f:
        f.write("<html><head><style>")
        f.write("table { border-collapse: collapse; width: 100%; }")
        f.write("th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }")
        f.write("img { max-width: 100px; display: block; margin: 0 auto; }")
        f.write(".correct { border: 3px solid green; }")
        f.write(".wrong { border: 3px solid red; }")
        f.write("</style></head><body>")
        f.write("<h2>CLIP Prediction Summary</h2>")
        f.write("<table>")
        f.write("<tr><th>Class</th><th>Images</th><th>Most Mistaken For</th><th>Second Most Mistaken For</th><th>Correct</th><th>Accuracy</th></tr>")

        for cls, entries in sorted(class_to_images.items()):
            correct_count = 0
            total = len(entries)
            incorrect_labels = []

            img_cells = ""
            for image_name, info in entries:
                correct = info["predicted_class"] == info["ground_truth_class"]
                predicted_label = info["predicted_class"]

                if correct:
                    correct_count += 1
                    img_cells += f'''<td class="correct"><img src="{os.path.join(image_dir, image_name)}" alt="{image_name}"><br>{image_name}<br>
                        <span style="color:green;"><b>{predicted_label}</td>'''
                else:
                    img_cells += f'''<td class="wrong"><img src="{os.path.join(image_dir, image_name)}" alt="{image_name}"><br>{image_name}<br>
                        <span style="color:red;"><b>{predicted_label}</td>'''
                    incorrect_labels.append(info["predicted_class"])

            common_mistakes = Counter(incorrect_labels).most_common(2)
            mistake_1 = common_mistakes[0][0] if len(common_mistakes) > 0 else "-"
            mistake_2 = common_mistakes[1][0] if len(common_mistakes) > 1 else "-"
            true_positives = confusion_matrix[cls][cls]
            false_positives = sum(confusion_matrix[cls].values()) - true_positives
            false_negatives = sum(confusion_matrix[other_cls][cls] for other_cls in confusion_matrix if other_cls != cls)

            accuracy = (correct_count / total) * 100
            precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
            recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            f.write(f"<tr>")
            f.write(f"<td>{cls}</td>")
            f.write(f"<td><table><tr>{img_cells}</tr></table></td>")
            f.write(f"<td>{mistake_1}</td>")
            f.write(f"<td>{mistake_2}</td>")
            f.write(f"<td>{correct_count}/{total}</td>")
            f.write(f"<td>{accuracy:.2f}%</td>")
            f.write(f"<td>Precision: {precision:.2f}%, Recall: {recall:.2f}%</td>")
            f.write(f"<td>F1 Score: {f1_score:.2f}</td>")
            f.write("</tr>")

        f.write("</table></body></html>")

    print(f"✅ Summary saved to {summarised_results_path}") 
    print("✅ Precision: {:.2f}%".format(sum(precision_list) / len(precision_list)))
    print("✅ Recall: {:.2f}%".format(sum(recall_list) / len(recall_list)))
    print("✅ F1 Score: {:.2f}".format(sum(f1_score_list) / len(f1_score_list)))


def create_detailed_html(results, image_dir, output_dir="outputs"):
    clip_detailed_results_path = os.path.join(output_dir, "clip_detailed_results.html")

    with open(clip_detailed_results_path, "w") as f:
        f.write("<html><body><h2>CLIP Prediction Results</h2>\n")
        for image_name, info in results.items():
            image_path = os.path.join(image_dir, image_name)
            f.write(f'<div style="margin-bottom:20px;">\n')
            f.write(f'<img src="{os.path.join(image_dir, image_name)}" width="150"><br>\n')
            f.write(f"<b>Image:</b> {image_name}<br>\n")
            f.write(f"<b>Ground truth:</b> {info['ground_truth_class']}<br>\n")
            # Highlight predicted label in green if correct
            if info['predicted_class'] == info['ground_truth_class']:
                f.write(f"<b>Predicted:</b> <span style='color:green;'>{info['predicted_class']}</span><br>\n")
            else:
                f.write(f"<b>Predicted:</b> <span style='color:red;'>{info['predicted_class']}</span><br>\n")
            f.write("<b>Top 3:</b><ul>\n")
            for cls, prob in info['top_5_probs'].items():
                f.write(f"<li>{cls} ({prob*100:.1f}%)</li>\n")
            f.write("</ul></div>\n")
        f.write("</body></html>")
    print("✅ Results saved to clip_detailed_results.html")


def create_output_htmls(results, image_dir):

    image_dir = os.path.join("")
    output_dir = "outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # Save predictions
    with open("outputs/clip_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    
    create_detailed_html(results, image_dir, output_dir)
    create_summarised_html(results, image_dir, output_dir)

class image_test_dataset(Dataset):
    def __init__(self, image_dir, class_name_to_idx, preprocess):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        self.image_files.sort()
        self.class_name_to_idx = class_name_to_idx
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        class_name = " ".join(image_name.split("_")[:-1]) 
        class_name_idx = self.class_name_to_idx[class_name]
        return self.preprocess(image), class_name_idx
    

def load_model_and_weights(device):
    """
    Load the CLIP model and its fine-tuned weights.
    """

    model, preprocess = clip.load("RN50", device=device)
    num_classes = len(class_name_promt_map)
    checkpoint = torch.load("fine_tuned_clip.pth")
    model.visual.attnpool.load_state_dict(checkpoint['attnpool_state_dict'])

    print("✅ Weights loaded successfully!")

    return model, preprocess


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_model_and_weights(device)
class_names = sorted(class_name_promt_map.keys())
text_prompts = [". ".join(class_name_promt_map[name]) for name in class_names]
text_inputs = clip.tokenize(text_prompts, truncate=True).to(device)
class_name_to_idx = {name: idx for idx, name in enumerate(sorted(class_name_promt_map.keys()))}
image_dir = ""
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
test_dataset = image_test_dataset(image_dir, class_name_to_idx, preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
results = {}
no_correct_predictions = 0
text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)


# Calculate similarity for each image
# for images, class_idx in tqdm(test_loader):
image_files.sort()

for image_name in image_files:
    print("Processing image...", image_name)
    class_name = " ".join(image_name.split("_")[:-1])

    with torch.no_grad():
        image_path = os.path.join(image_dir, image_name)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)


        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.squeeze().tolist()

        best_idx = int(torch.argmax(similarity))
        predicted_class = class_names[best_idx]



        top5 = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)[:5]
        results[image_name] = {
            "predicted_class": predicted_class,
            "ground_truth_class": class_name,
            # "probabilities": probs[i],
            "top_5_probs": dict(top5),
        }
        if results[image_name]["predicted_class"] == results[image_name]["ground_truth_class"]:
            print(f"✅ Correct prediction for {image_name}: {results[image_name]['predicted_class']}")
            no_correct_predictions += 1


# Print results
print("Results:")
for image_name, info in results.items():
    print(f"Image: {image_name}, Predicted: {info['predicted_class']}, Ground Truth: {info['ground_truth_class']}")
print("Total images:", len(test_dataset))
print("Correct predictions:", no_correct_predictions)
print("Accuracy:", no_correct_predictions / len(test_dataset))

# Correct predictions in percentage
print(f"✅ Correct predictions: {no_correct_predictions}/{len(test_dataset)} ({(no_correct_predictions /len(test_dataset)) * 100:.2f}%)")

no_correct_top3_predictions = 0
no_correct_top5_predictions = 0
for image_name, info in results.items():
    top_3_classes = sorted(info['top_5_probs'].items(), key=lambda x: x[1], reverse=True)[:3]
    if info['ground_truth_class'] in dict(top_3_classes):
        no_correct_top3_predictions += 1
    if info['ground_truth_class'] in info['top_5_probs']:
        no_correct_top5_predictions += 1
print(f"✅ Correct top 3 predictions: {no_correct_top3_predictions}/{len(test_dataset)} ({(no_correct_top3_predictions / len(test_dataset)) * 100:.2f}%)")
print(f"✅ Correct top 5 predictions: {no_correct_top5_predictions}/{len(test_dataset)} ({(no_correct_top5_predictions / len(test_dataset)) * 100:.2f}%)")

# mean per class accuracy
class_correct = defaultdict(int)
class_total = defaultdict(int)
for image_name, info in results.items():
    class_total[info['ground_truth_class']] += 1
    if info['predicted_class'] == info['ground_truth_class']:
        class_correct[info['ground_truth_class']] += 1  
mean_per_class_accuracy = sum((class_correct[c] / class_total[c]) for c in class_total) / len(class_total)
print(f"✅ Mean per class accuracy: {mean_per_class_accuracy * 100:.2f}%")

create_output_htmls(results, image_dir)
