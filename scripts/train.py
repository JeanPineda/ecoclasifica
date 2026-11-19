"""Train and evaluate EcoClasifica with a fixed, external test partition."""
import argparse, csv, hashlib, json, random, urllib.request, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["glass", "metal", "plastic"]
NAMES = ["Vidrio", "Metal", "Plástico"]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument("--archive", default=str(ROOT / "data" / "trashnet.zip"))
args = parser.parse_args()
archive = Path(args.archive)
archive.parent.mkdir(parents=True, exist_ok=True)
if not archive.exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip", archive)
data = ROOT / "data" / "images"
data.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        parts = Path(item.filename).parts
        if len(parts) >= 2 and parts[-2] in CLASSES and parts[-1].lower().endswith((".jpg", ".png", ".jpeg")):
            (data / parts[-1]).write_bytes(z.read(item))
records = []
for path in sorted(data.glob("*")):
    cls = next((c for c in CLASSES if path.name.startswith(c)), None)
    if cls is None:
        continue
    with Image.open(path) as im:
        im = im.convert("RGB")
        sha = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        gray = np.asarray(im.resize((9, 8)).convert("L"))
        dhash = int("".join("1" if b else "0" for b in (gray[:, 1:] > gray[:, :-1]).flat), 2)
    records.append({"path": path, "label": CLASSES.index(cls), "sha": sha, "hash": dhash})
# Group exact and extremely close perceptual matches before assigning partitions.
parent = list(range(len(records)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
for i in range(len(records)):
    for j in range(i):
        if records[i]["sha"] == records[j]["sha"] or (records[i]["hash"] ^ records[j]["hash"]).bit_count() <= 1:
            parent[find(i)] = find(j)
groups = {}
for i, r in enumerate(records):
    groups.setdefault(find(i), []).append(r)
excluded, cleaned, group_ids, labels = [], [], [], []
for gid, rows in groups.items():
    if len({r["label"] for r in rows}) > 1:
        excluded.extend({"file": r["path"].name, "reason": "conflicting_labels_in_similarity_group"} for r in rows)
        continue
    seen = set()
    keep = []
    for r in rows:
        if r["sha"] in seen:
            excluded.append({"file": r["path"].name, "reason": "exact_duplicate"})
        else:
            seen.add(r["sha"])
            keep.append(r)
    group_ids.append(gid)
    labels.append(rows[0]["label"])
    for r in keep:
        r["group"] = gid
        cleaned.append(r)
train_ids, hold_ids = train_test_split(group_ids, test_size=.30, random_state=SEED, stratify=labels)
label_map = dict(zip(group_ids, labels))
val_ids, test_ids = train_test_split(hold_ids, test_size=.5, random_state=SEED, stratify=[label_map[g] for g in hold_ids])
sets = {"train": set(train_ids), "validation": set(val_ids), "test": set(test_ids)}
for r in cleaned:
    r["split"] = next(k for k, ids in sets.items() if r["group"] in ids)
reports = ROOT / "reports"
models = ROOT / "models"
reports.mkdir(exist_ok=True)
models.mkdir(exist_ok=True)
with (reports / "manifest.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "class", "split", "group", "sha256_pixels"])
    writer.writerows([r["path"].name, CLASSES[r["label"]], r["split"], r["group"], r["sha"]] for r in cleaned)
(reports / "exclusions.json").write_text(json.dumps(excluded, indent=2))
weights = MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()
class Photos(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(r["path"]) as im:
            x = transform(ImageOps.exif_transpose(im).convert("RGB"))
        return x, r["label"]
backbone = mobilenet_v3_small(weights=weights)
backbone.classifier = nn.Identity()
backbone.eval()
def features(rows):
    xs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(Photos(rows), batch_size=32):
            xs.append(backbone(x))
            ys.append(y)
    return torch.cat(xs), torch.cat(ys)
partitions = {k: [r for r in cleaned if r["split"] == k] for k in sets}
print("Extracting train and validation features", flush=True)
train_x, train_y = features(partitions["train"])
val_x, val_y = features(partitions["validation"])
head = nn.Linear(train_x.shape[1], 3)
optimizer = torch.optim.Adam(head.parameters(), lr=.001, weight_decay=.001)
loss_fn = nn.CrossEntropyLoss()
history, best, stale = [], -1, 0
for epoch in range(100):
    head.train()
    order = torch.randperm(len(train_x))
    for batch in order.split(16):
        optimizer.zero_grad()
        loss = loss_fn(head(train_x[batch]), train_y[batch])
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        from sklearn.metrics import f1_score
        pred = head(val_x).argmax(1).numpy()
        f1 = f1_score(val_y.numpy(), pred, average="macro", zero_division=0)
        row = {"epoch": epoch + 1, "train_loss": loss_fn(head(train_x), train_y).item(),
               "validation_loss": loss_fn(head(val_x), val_y).item(), "validation_f1_macro": f1}
    history.append(row)
    if f1 > best:
        best, stale = f1, 0
        torch.save(head.state_dict(), models / "head.pt")
        best_epoch = epoch + 1
    else:
        stale += 1
    if stale >= 15:
        break
head.load_state_dict(torch.load(models / "head.pt", weights_only=True))
model = nn.Sequential(backbone, head, nn.Softmax(dim=1)).eval()
torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(models / "model.onnx"),
                  input_names=["images"], output_names=["probabilities"],
                  dynamic_axes={"images": {0: "batch"}, "probabilities": {0: "batch"}},
                  opset_version=17, dynamo=False)
import onnxruntime as ort
session = ort.InferenceSession(str(models / "model.onnx"), providers=["CPUExecutionProvider"])
# Evaluate the final ONNX export, once selected on validation.
test_rows = partitions["test"]
probabilities, actual, filenames = [], [], []
max_delta = 0.
with torch.no_grad():
    for x, y in DataLoader(Photos(test_rows), batch_size=32):
        probabilities.extend(session.run(None, {"images": x.numpy()})[0])
        actual.extend(y.numpy())
# Numerical conversion check uses validation images.
with torch.no_grad():
    check = next(iter(DataLoader(Photos(partitions["validation"]), batch_size=8)))[0]
    max_delta = float(np.abs(model(check).numpy() - session.run(None, {"images": check.numpy()})[0]).max())
probs = np.array(probabilities)
actual = np.array(actual)
pred = probs.argmax(1)
report = classification_report(actual, pred, labels=[0,1,2], target_names=NAMES, output_dict=True, zero_division=0)
aucs = {name: float(roc_auc_score(actual == i, probs[:, i])) for i, name in enumerate(NAMES)}
summary = {"seed": SEED, "source_images": len(records), "usable_images": len(cleaned), "excluded_images": len(excluded),
           "split_counts": {k: {name: sum(r["label"] == i for r in rows) for i,name in enumerate(NAMES)} for k,rows in partitions.items()},
           "best_epoch": best_epoch, "epochs_executed": len(history), "validation_f1_macro": best,
           "test_report": report, "auc": aucs, "auc_macro": float(np.mean(list(aucs.values()))),
           "onnx_max_absolute_difference": max_delta, "confusion_matrix": confusion_matrix(actual,pred).tolist(),
           "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
(reports / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
(reports / "history.json").write_text(json.dumps(history, indent=2))
with (reports / "test_predictions.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "actual", "predicted", *CLASSES])
    for row,y,yp,ps in zip(test_rows,actual,pred,probs):
        writer.writerow([row["path"].name,CLASSES[y],CLASSES[yp],*ps])
plt.figure(figsize=(6,5))
cm=confusion_matrix(actual,pred)
plt.imshow(cm,cmap="Blues")
for i in range(3):
    for j in range(3):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
plt.xticks(range(3),NAMES);plt.yticks(range(3),NAMES)
plt.xlabel("Predicción");plt.ylabel("Clase real");plt.title("Matriz de confusión · prueba externa")
plt.tight_layout();plt.savefig(reports/"confusion.png",dpi=160);plt.close()
plt.figure(figsize=(6,5))
for i,name in enumerate(NAMES):
    fpr,tpr,_=roc_curve(actual==i,probs[:,i])
    plt.plot(fpr,tpr,label=f"{name} · AUC {aucs[name]:.3f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("Tasa de falsos positivos");plt.ylabel("Tasa de verdaderos positivos")
plt.title("ROC uno contra el resto · prueba externa");plt.legend()
plt.tight_layout();plt.savefig(reports/"roc.png",dpi=160);plt.close()
plt.figure(figsize=(7,3))
plt.plot([h["epoch"] for h in history],[h["train_loss"] for h in history],label="Entrenamiento")
plt.plot([h["epoch"] for h in history],[h["validation_loss"] for h in history],label="Validación")
plt.xlabel("Época");plt.ylabel("Entropía cruzada");plt.legend();plt.tight_layout()
plt.savefig(reports/"learning.png",dpi=160);plt.close()
metadata={"classes":CLASSES,"display_names":NAMES,"version":"ecoclasifica-v1",
          "resize_shorter":256,"crop":224,"mean":[.485,.456,.406],"std":[.229,.224,.225]}
(models/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
samples=ROOT/"sample_images"
samples.mkdir(exist_ok=True)
for i in range(3):
    for row in [r for r in test_rows if r["label"]==i][:3]:
        import shutil
        shutil.copy(row["path"],samples/row["path"].name)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
