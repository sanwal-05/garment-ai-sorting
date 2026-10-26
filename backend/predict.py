import argparse, json, joblib, numpy as np, torch
import torch.nn.functional as Fn
from PIL import Image
from transformers import AutoModel, CLIPModel
import io

MEAN = {"dino": (0.485, 0.456, 0.406), "clip": (0.4815, 0.4578, 0.4082)}
STD  = {"dino": (0.229, 0.224, 0.225), "clip": (0.2686, 0.2613, 0.2758)}

class Predictor:
    def __init__(self, bundle_path="garment_bundle.joblib", device=None):
        self.b = joblib.load(bundle_path)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.models = {}
        for short, (hf, size) in self.b["emb_cfg"].items():
            kind = "clip" if "clip" in hf else "dino"
            m = (CLIPModel if kind == "clip" else AutoModel).from_pretrained(hf, torch_dtype=self.dt).to(self.dev).eval()
            self.models[short] = (m, kind, size)

    @torch.no_grad()
    def _embed(self, short, imgs):
        m, kind, size = self.models[short]
        mean = torch.tensor(MEAN[kind], device=self.dev).view(1, 3, 1, 1); std = torch.tensor(STD[kind], device=self.dev).view(1, 3, 1, 1)
        x = torch.from_numpy(imgs).to(self.dev).permute(0, 3, 1, 2).float() / 255
        x = Fn.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - mean) / std).to(self.dt); fs = []
        for xx in (x, x.flip(-1)):
            if kind == "clip":
                f = m.get_image_features(pixel_values=xx)
                if not torch.is_tensor(f): f = f.pooler_output
            else:
                h = m(pixel_values=xx).last_hidden_state
                f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
            fs.append(f.float())
        return ((fs[0] + fs[1]) / 2).cpu().numpy()

    @staticmethod
    def _load(src):
        im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        return np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)

    def predict(self, front_paths, back_paths, types=None):
        fr = np.stack([self._load(p) for p in front_paths]); bk = np.stack([self._load(p) for p in back_paths])
        E = {}
        for s in self.models: E[(s, "front")] = self._embed(s, fr); E[(s, "back")] = self._embed(s, bk)
        types = types or [""] * len(fr)
        tid = [self.b["type_map"].get(str(t).strip().title(), 0) for t in types]
        oh = np.eye(26, dtype=np.float32)[tid]
        feats = {name: np.c_[pca.transform(sc.transform(np.concatenate([E[k] for k in keys], 1))).astype(np.float32), oh]
                 for name, (keys, sc, pca) in self.b["tf"].items()}
        out = {}
        for t in self.b["tasks"]:
            Ps = []
            for (tt, fs, hn), mdl in self.b["models"].items():
                if tt != t: continue
                X = feats[fs]
                if isinstance(mdl, list):
                    c1 = mdl[0].predict_proba(X)[:, 1]; c2 = mdl[1].predict_proba(X)[:, 1]
                    P = np.c_[1 - c1, np.clip(c1 - c2, 1e-4, None), c2]; P = P / P.sum(1, keepdims=True)
                else: P = mdl.predict_proba(X)
                Ps.append(P)
            out[t] = np.mean(Ps, 0)
        return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("front"); ap.add_argument("back")
    ap.add_argument("--type", default=""); ap.add_argument("--bundle", default="garment_bundle.joblib")
    a = ap.parse_args(); r = Predictor(a.bundle).predict([a.front], [a.back], [a.type])
    print(json.dumps({"condition_probs[low(1-2), mid(3), high(4-5)]": r["condition"][0].round(3).tolist(),
                      "pilling_probs[1-3, 4, 5]": r["pilling"][0].round(3).tolist(),
                      "p_stains": round(float(r["stains"][0, 1]), 3), "p_holes": round(float(r["holes"][0, 1]), 3)}, indent=2))
