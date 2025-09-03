from prompts import class_name_prompt_map
import os
import random

image_dir = 'local/latest/China/cn_test'
class_names = sorted(list(class_name_prompt_map.keys()))
num_classes = len(class_names)

def random_classifier(image_path):
    # Randomly select a class index
    random.seed(42)  # Initialize the random number generator
    class_index = random.randint(0, num_classes - 1)
    class_name = class_names[class_index]
    return class_name

if __name__ == "__main__":
    classwise_accuracy = {class_name: 0 for class_name in class_names}
    images_per_class = {class_name: 0 for class_name in class_names}
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        predicted_class = random_classifier(image_path)
        image_name = os.path.splitext(image_file)[0]
        true_class = " ".join(image_name.split("_")[:-1])
        images_per_class[true_class] += 1
        if predicted_class == true_class:
            classwise_accuracy[true_class] += 1
    for class_name, correct_count in classwise_accuracy.items():
        print(f"Class: {class_name}, Correct Predictions: {correct_count}")
        classwise_accuracy[class_name] = correct_count / images_per_class[class_name] if images_per_class[class_name] > 0 else 0
    overall_accuracy = sum(classwise_accuracy.values()) / num_classes


    print(f"Overall Accuracy: {overall_accuracy:.4f}")