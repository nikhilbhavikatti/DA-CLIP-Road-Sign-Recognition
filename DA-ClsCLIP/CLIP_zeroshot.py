import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Optional

def standardize_label(label):
    return label.lower().replace(' ', '').replace('_', '')

class CLIPzeroshot:
    def __init__(
        self,
        dataset_folder: str,
        groundtruth_path: str,
        prompts_path: str,
        save_folder: str,
        model_name: str = "openai/clip-vit-base-patch32",
        device=None,
        finetuned_ckpt: Optional[str] = None,
    ):
        self.dataset_folder = dataset_folder
        self.groundtruth_path = groundtruth_path
        self.prompts_path = prompts_path
        self.save_folder = save_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # === 1. 加载微调后的视觉模型参数（如果有） ===
        if finetuned_ckpt:
            ckpt = torch.load(finetuned_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            # 注意: 不加载 text_features，也不加载 categories！

        # === 2. 加载当前 prompt 文件、动态编码 text_features ===
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            self.prompt_data = json.load(f)   # {label: [prompt1, ...]}
        self.categories = list(self.prompt_data.keys())
        self.prompt_variations = list(self.prompt_data.values())
        self.text_features, self.prompt_to_cat = self._init_text_features()

    def _init_text_features(self):
        all_prompts = [p for plist in self.prompt_variations for p in plist]
        prompt_to_cat = []
        for idx, plist in enumerate(self.prompt_variations):
            prompt_to_cat += [idx] * len(plist)
        inputs = self.processor(
            text=all_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats, prompt_to_cat

    def classify_image(self, img: Image.Image, top_k=5, return_all=False):
        inputs = self.processor(images=img, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * img_feat @ self.text_features.T
            probs = logits.softmax(dim=-1)[0].cpu().numpy()
        # 聚合到类别
        cat_probs = np.zeros(len(self.categories))
        for i, p in enumerate(probs):
            cat_probs[self.prompt_to_cat[i]] += p
        cat_probs /= cat_probs.sum()
        if return_all:
            idxs = np.argsort(-cat_probs)
        else:
            idxs = np.argpartition(cat_probs, -top_k)[-top_k:]
            idxs = idxs[np.argsort(-cat_probs[idxs])]
        return [(self.categories[i], float(cat_probs[i])) for i in idxs]
    
    def classify_batch(self, imgs: list, top_k=5, return_all=False):
        return [self.classify_image(img, top_k=top_k, return_all=return_all) for img in imgs]
    
    def evaluate_directory(self, batch_size=16, top_k=5):
        # 读取 groundtruth
        with open(self.groundtruth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        files = [f for f in os.listdir(self.dataset_folder) if f.lower().endswith(".png")]
        rows = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            imgs  = []
            for fn in batch:
                img = Image.open(os.path.join(self.dataset_folder, fn)).convert("RGB")
                imgs.append(img)
            preds_batch = self.classify_batch(imgs, top_k=top_k, return_all=True)
            for fn, preds in zip(batch, preds_batch):
                true = ground_truth.get(fn, "Unknown")
                labels = [x[0] for x in preds[:5]]
                scores = [x[1] for x in preds[:5]]
                if len(labels) < 5:
                    labels += [""] * (5 - len(labels))
                    scores += [0.0] * (5 - len(scores))
                row = {
                    "Image": fn,
                    "True Label": true
                }
                for i in range(5):
                    row[f"Top{i+1}_Label"] = labels[i]
                    row[f"Top{i+1}_Prob"] = scores[i]
                found = -1
                for k in range(1, 6):
                    if row[f"Top{k}_Label"] == true:
                        found = k
                        break
                row["Rank"] = found
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.save_folder, "per_image_top5.csv"), index=False)
        return df

    def overall_macro_metrics(self, df_results):
        y_true = df_results["True Label"].values
        y_pred = df_results["Top1_Label"].values
        acc = accuracy_score(y_true, y_pred)
        label_names = self.categories
        p, r, f1, s = precision_recall_fscore_support(
            y_true, y_pred, labels=label_names, average='macro', zero_division=0)
        overall_metrics = {
            "Accuracy": [acc],
            "Macro Precision": [p],
            "Macro Recall": [r],
            "Macro F1": [f1],
            "Support": [len(y_true)]
        }
        df_overall_metrics = pd.DataFrame(overall_metrics)
        df_overall_metrics.to_csv(os.path.join(self.save_folder, "overall_macro_metrics.csv"), index=False)
        return df_overall_metrics

    def overall_top5_accuracy(self, df_results):
        Ks = [1, 2, 3, 4, 5]
        overall_acc = {}
        for k in Ks:
            hit = 0
            for i, row in df_results.iterrows():
                found = False
                for ki in range(1, k+1):
                    if row[f"Top{ki}_Label"] == row["True Label"]:
                        found = True
                        break
                if found:
                    hit += 1
            overall_acc[f"Top-{k}"] = hit / len(df_results)
        overall_acc_df = pd.DataFrame(list(overall_acc.items()), columns=["Top-K", "Accuracy"])
        overall_acc_df.to_csv(os.path.join(self.save_folder, "overall_top5_accuracy.csv"), index=False)
        return overall_acc_df

    def per_class_macro_metrics(self, df_results):
        y_true = df_results["True Label"].values
        y_pred = df_results["Top1_Label"].values
        label_names = self.categories
        p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
            y_true, y_pred, labels=label_names, average=None, zero_division=0)
        df_per_class_metrics = pd.DataFrame({
            "Class": label_names,
            "Precision": p_c,
            "Recall": r_c,
            "F1": f1_c,
            "Support": s_c
        })
        df_per_class_metrics.to_csv(os.path.join(self.save_folder, "per_class_macro_metrics.csv"), index=False)
        return df_per_class_metrics

    def per_class_top5_accuracy(self, df_results):
        label_names = self.categories
        Ks = [1, 2, 3, 4, 5]
        per_class_acc = []
        for class_name in label_names:
            row = [class_name]
            mask = df_results["True Label"] == class_name
            df_sub = df_results[mask]
            n = len(df_sub)
            for k in Ks:
                hit = 0
                for _, r in df_sub.iterrows():
                    found = False
                    for ki in range(1, k+1):
                        if r[f"Top{ki}_Label"] == r["True Label"]:
                            found = True
                            break
                    if found:
                        hit += 1
                acc = hit / n if n > 0 else 0.0
                row.append(acc)
            per_class_acc.append(row)
        header = ["Class"] + [f"Top-{k} Acc" for k in Ks]
        df_per_class_acc = pd.DataFrame(per_class_acc, columns=header)
        df_per_class_acc.to_csv(os.path.join(self.save_folder, "per_class_top5_accuracy.csv"), index=False)
        return df_per_class_acc
    
    def zeroshot(self):
        df_de = self.evaluate_directory(batch_size=16, top_k=5)
        self.overall_macro_metrics(df_de)
        self.overall_top5_accuracy(df_de)
        self.per_class_macro_metrics(df_de)
        self.per_class_top5_accuracy(df_de)
    
def test3(
    rsfolder,
    model=None,
    prompt_path=None,           # 新增参数：手动指定prompt
    domains=("cn", "de", "in"),
):
    for domain in domains:
        print(f"Predicting {domain} ...")
        if prompt_path is not None:
            zargs = dict(
                dataset_folder=f"test_{domain}",
                groundtruth_path=f"test_{domain}.json",
                prompts_path=prompt_path,     # 统一用同一个指定的prompt
                save_folder=os.path.join(rsfolder, domain)
            )
        else:
            zargs = dict(
                dataset_folder=f"test_{domain}",
                groundtruth_path=f"test_{domain}.json",
                prompts_path=f"prompt_{domain}.json",     # 用各自的prompt
                save_folder=os.path.join(rsfolder, domain)
            )
        if model is not None:
            zargs['finetuned_ckpt'] = model
        zero = CLIPzeroshot(**zargs)
        zero.zeroshot()
        print(f"\tSaved results in {os.path.join(rsfolder, domain)}")





# 用法示例
if __name__ == "__main__":
    # test3(
    #     rsfolder="RSzeroshot",
    #     model=None,                       # 不传模型路径就是 zero-shot
    #     domains=("cn", "de", "in"),       # 需要评测哪些test集
    #     test_prefix="test",               # 文件夹名
    #     gt_prefix="test",                 # 标注文件名
    #     prompt_prefix="prompt"            # prompt文件名
    # )

    test3(
        rsfolder="Rsfinetune_mx_cn",
        model="clip_finetune_mx_cn.pth",
        # prompt_path="prompt_cn.json",
        domains=("cn", "de", "in"),
    )

    test3(
        rsfolder="Rsfinetune_mx_de",
        model="clip_finetune_mx_de.pth",
        # prompt_path="prompt_de.json",
        domains=("cn", "de", "in"),
    )

    test3(
        rsfolder="Rsfinetune_mx_in",
        model="clip_finetune_mx_in.pth",
        # prompt_path="prompt_in.json",
        domains=("cn", "de", "in"),
    )

    test3(
        rsfolder="Rsfinetune_ai_cn",
        model="clip_finetune_ai_cn.pth",
        # prompt_path="prompt_cn.json",
        domains=("cn", "de", "in"),
    )

    test3(
        rsfolder="Rsfinetune_ai_de",
        model="clip_finetune_ai_de.pth",
        # prompt_path="prompt_de.json",
        domains=("cn", "de", "in"),
    )

    test3(
        rsfolder="Rsfinetune_ai_in",
        model="clip_finetune_ai_in.pth",
        # prompt_path="prompt_in.json",
        domains=("cn", "de", "in"),
    )



    print("Done.")