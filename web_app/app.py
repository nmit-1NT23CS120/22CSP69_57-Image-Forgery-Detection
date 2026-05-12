"""
=============================================================
Flask Web Application — Image Forgery Detection
=============================================================
Run: python web_app/app.py
Open: http://localhost:5000
=============================================================
"""

import os, sys, io, uuid, json, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from flask import (Flask, render_template, request,
                   jsonify, url_for)
from werkzeug.utils import secure_filename

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
UPLOAD_DIR = os.path.join(BASE_DIR, 'web_app', 'static', 'uploads')
RESULT_DIR = os.path.join(BASE_DIR, 'web_app', 'static', 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.append(BASE_DIR)
app    = Flask(__name__, template_folder='templates', static_folder='static')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

class ResNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.resnet50(weights=None)
        backbone_layers = list(b.children())[:-1]
        self.model = nn.Sequential(
            nn.Sequential(*backbone_layers),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        return self.model(x)
    def load_ckpt(self, state):
        remapped = {f'model.{k}': v for k, v in state.items()}
        self.load_state_dict(remapped, strict=False)

class EfficientNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.efficientnet_b4(weights=None)
        self.features   = b.features
        self.avgpool    = b.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(1792, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 2))
    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))

class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4*4, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3))
    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        k = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]],
                         dtype=x.dtype, device=x.device)
        f = F.conv2d(g, k, padding=1)
        f = torch.abs(f) / (torch.abs(f).max() + 1e-8)
        return self.fc(self.freq_cnn(f))

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.efficientnet_b4(weights=None)
        self.spatial_features = b.features
        self.spatial_pool     = b.avgpool
        self.freq_branch      = FrequencyBranch()
        self.classifier       = nn.Sequential(
            nn.Flatten(), nn.Linear(1792+256, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 2))
    def forward(self, x):
        sp = self.spatial_pool(self.spatial_features(x)).squeeze(-1).squeeze(-1)
        fr = self.freq_branch(x)
        return self.classifier(torch.cat([sp, fr], dim=1))

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_resnet():
    path = os.path.join(MODELS_DIR, 'resnet_baseline.pth')
    if not os.path.exists(path): print('  WARNING: resnet not found'); return None
    ck = torch.load(path, map_location=DEVICE)
    state = ck.get('model_state', ck)
    m = ResNetBaseline().to(DEVICE)
    m.load_ckpt(state)
    m.eval()
    print('  Loaded resnet_baseline.pth')
    return m

def load_model(cls, fname):
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path): print(f'  WARNING: {fname} not found'); return None
    ck = torch.load(path, map_location=DEVICE)
    state = ck.get('model_state', ck)
    m = cls().to(DEVICE)
    m.load_state_dict(state, strict=True)
    m.eval()
    print(f'  Loaded {fname}')
    return m

print('Loading models...')
MODELS = {
    'resnet'      : load_resnet(),
    'efficientnet': load_model(EfficientNetModel, 'efficientnet_model.pth'),
    'hybrid'      : load_model(HybridModel, 'hybrid_model.pth'),
}
print('All models loaded!')

def jpeg_compress_pil(pil_img, quality):
    buf = io.BytesIO()
    pil_img.convert('RGB').save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).copy()

def compress_facebook(img):
    w,h = img.size
    if max(w,h)>2048: s=2048/max(w,h); img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return jpeg_compress_pil(img, 75)

def compress_instagram(img):
    w,h = img.size
    if max(w,h)>1080: s=1080/max(w,h); img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)
    return jpeg_compress_pil(img, 78)

def compress_whatsapp(img):
    w,h = img.size
    if max(w,h)>1600: s=1600/max(w,h); img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)
    img = jpeg_compress_pil(img, 60)
    return jpeg_compress_pil(img, 55)

def compress_twitter(img):
    w,h = img.size
    if max(w,h)>1200: s=1200/max(w,h); img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)
    return jpeg_compress_pil(img, 82)

COMPRESSORS = {
    'facebook':compress_facebook, 'instagram':compress_instagram,
    'whatsapp':compress_whatsapp, 'twitter':compress_twitter,
}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.grads = None; self.acts = None
        target_layer.register_forward_hook(
            lambda m,i,o: setattr(self,'acts',o.detach()))
        target_layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self,'grads',go[0].detach()))
    def generate(self, tensor):
        self.model.eval()
        out = self.model(tensor)
        ci  = out.argmax(1).item()
        self.model.zero_grad()
        oh  = torch.zeros_like(out); oh[0][ci] = 1
        out.backward(gradient=oh, retain_graph=True)
        pg  = self.grads.mean(dim=[0,2,3])
        a   = self.acts[0].clone()
        for i in range(a.shape[0]): a[i] *= pg[i]
        hm  = a.mean(0).cpu().numpy()
        hm  = np.maximum(hm, 0)
        if hm.max() > 0: hm /= hm.max()
        return hm, ci

def make_heatmap(img_np, heatmap):
    h,w = img_np.shape[:2]
    hm  = cv2.resize(heatmap, (w,h))
    hmu = np.uint8(255*hm)
    hmc = cv2.cvtColor(cv2.applyColorMap(hmu, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np.astype(np.uint8),0.55,hmc,0.45,0)

def predict(model_key, pil_img, generate_heatmap=False):
    model = MODELS.get(model_key)
    if model is None: return None
    orig_np = np.array(pil_img.resize((224,224)))
    tensor  = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    fp = float(probs[1]); ap = float(probs[0])
    if fp < 0.3:   trust, color = 'LOW RISK',   '#22c55e'
    elif fp < 0.7: trust, color = 'SUSPICIOUS', '#f59e0b'
    else:          trust, color = 'HIGH RISK',  '#ef4444'
    result = {
        'predicted'  : 'FORGED' if fp>0.5 else 'AUTHENTIC',
        'forged_prob': round(fp*100, 1),
        'auth_prob'  : round(ap*100, 1),
        'trust'      : trust,
        'trust_color': color,
        'heatmap_url': None,
    }
    if generate_heatmap and model_key == 'hybrid':
        try:
            target = model.spatial_features[-1]
            gcam   = GradCAM(model, target)
            t2     = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            t2.requires_grad_(True)
            hm, _  = gcam.generate(t2)
            overlay  = make_heatmap(orig_np, hm)
            hm_name  = f'heatmap_{uuid.uuid4().hex[:8]}.jpg'
            hm_path  = os.path.join(RESULT_DIR, hm_name)
            Image.fromarray(overlay.astype(np.uint8)).save(hm_path)
            result['heatmap_url'] = url_for('static', filename=f'results/{hm_name}')
        except Exception as e:
            print(f'GradCAM error: {e}')
    return result

def test_all_platforms(pil_img):
    results = {}
    model_labels = {'resnet':'ResNet50','efficientnet':'EfficientNet-B4','hybrid':'Hybrid (Ours)'}
    for platform, compressor in COMPRESSORS.items():
        comp_img = compressor(pil_img.copy())
        pres = {}
        for mkey, mname in model_labels.items():
            if MODELS[mkey] is None: continue
            r = predict(mkey, comp_img)
            if r:
                pres[mkey] = {
                    'name'      : mname,
                    'predicted' : r['predicted'],
                    'confidence': r['forged_prob'],
                }
        results[platform] = pres
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    f   = request.files['image']
    ext = os.path.splitext(secure_filename(f.filename))[1].lower()
    if ext not in {'.jpg','.jpeg','.png','.bmp','.webp'}:
        return jsonify({'error': 'Invalid file type'}), 400
    fname    = f'upload_{uuid.uuid4().hex[:8]}{ext}'
    img_path = os.path.join(UPLOAD_DIR, fname)
    f.save(img_path)
    pil_img  = Image.open(img_path).convert('RGB')
    img_url  = url_for('static', filename=f'uploads/{fname}')
    main_result = predict('hybrid', pil_img, generate_heatmap=True)
    if main_result is None:
        return jsonify({'error': 'Model not loaded'}), 500
    main_result['img_url'] = img_url
    model_comparison = {}
    for mkey, mname in [('resnet','ResNet50'),('efficientnet','EfficientNet-B4'),('hybrid','Hybrid (Ours)')]:
        r = predict(mkey, pil_img)
        if r:
            model_comparison[mname] = {
                'predicted' : r['predicted'],
                'confidence': r['forged_prob'],
                'trust'     : r['trust'],
            }
    main_result['model_comparison'] = model_comparison
    main_result['platform_results'] = test_all_platforms(pil_img)
    return jsonify(main_result)

if __name__ == '__main__':
    print('\n' + '='*50)
    print('Image Forgery Detection — Web App')
    print('Open: http://localhost:5000')
    print('='*50 + '\n')
    app.run(debug=True, host='0.0.0.0', port=5000)
