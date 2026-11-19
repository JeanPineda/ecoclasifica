"""Verify real inference, preprocessing consistency, and recoverable input errors."""
import csv, io, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import app, preprocess
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights
transform=MobileNet_V3_Small_Weights.DEFAULT.transforms()
client=app.test_client()
assert client.get("/").status_code==200
assert client.get("/health").json["status"]=="ok"
assert client.post("/predict").status_code==400
assert client.post("/predict",data={"images":(io.BytesIO(b"invalid"),"bad.png")}).json["results"][0]["status"]=="error"
mixed=[(io.BytesIO(b"invalid"),"bad.png")]
files=list((ROOT/"sample_images").glob("*.jpg"))[:3]
for path in files:
    mixed.append((io.BytesIO(path.read_bytes()),path.name))
rows=client.post("/predict",data={"images":mixed}).json["results"]
assert [r["status"] for r in rows]==["error","ok","ok","ok"]
assert client.post("/predict",data={"images":[(io.BytesIO(b"x"),f"{i}.png") for i in range(11)]}).status_code==400
assert client.post("/predict",data={"images":(io.BytesIO(b"x"*(5*1024*1024+1)),"big.jpg")}).json["results"][0]["status"]=="error"
predictions=list(csv.DictReader((ROOT/"reports/test_predictions.csv").open()))
delta=0.
for row in predictions:
    path=ROOT/"data/images"/row["file"]
    with Image.open(path) as im:
        delta=max(delta,float(np.abs(preprocess(im)[0]-transform(im.convert("RGB")).numpy()).max()))
    result=client.post("/predict",data={"images":(io.BytesIO(path.read_bytes()),path.name)}).json["results"][0]
    assert result["status"]=="ok"
    assert result["predicted_class"]==row["predicted"]
    for name,key in [("Vidrio","glass"),("Metal","metal"),("Plástico","plastic")]:
        assert abs(result["scores"][name]-float(row[key]))<.0001
assert delta<.00001
result={"status":"passed","test_images_rechecked":len(predictions),"preprocessing_max_difference":delta,
        "checks":["home","health","empty request","invalid image","mixed batch","batch limit","file size limit","208 predictions match evaluation"]}
(ROOT/"reports/verification.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
