# Architectural Blueprint: Self-Trained Lightweight NSFW Image Classifier

This design document specifies the architecture, dataset collection, training recipe, and quantization pipeline for replacing the external `nudenet` dependency with a 100% team-trained lightweight computer vision model (<8MB) capable of running client-side in the Chrome extension (via ONNX Runtime Web / WebGPU) or on low-resource backend containers.

---

## 1. Problem & Motivation

The default hackathon prototype relies on `NudeNet`, which has several deployment drawbacks:
1. **Third-Party Dependency**: Violates the "majority self-trained" hackathon goal.
2. **Cold Boot Latency**: Downloads heavy multi-megabyte weights on first boot.
3. **High Resource Consumption**: Heavy memory footprint (>400MB RAM) not suitable for sub-1GB RAM container tiers (Fly.io free/hobby tier).

---

## 2. Model Architecture: MobileNetV3-Small-NSFW

We select **MobileNetV3-Small** (with hard-swish activations and squeeze-and-excitation blocks) as the backbone:
- **Input Resolution**: $224 \times 224 \times 3$ RGB.
- **Parameters**: ~1.5M parameters.
- **Inference Latency**: ~8ms on CPU, ~2ms with WebGPU.
- **Output Classes (3-way)**:
  1. `neutral` (everyday photos, landscapes, pets, food, memes)
  2. `nsfw_graphic` (nudity, explicit content, pornography)
  3. `nsfw_gore` (blood, severe violence, graphic bodily injury)

```
[Input: 224x224x3]
       │
       ▼
[MobileNetV3-Small Backbone (Feature Extractor)]
       │ (Feature map: 7x7x576)
       ▼
[Global Average Pooling]
       │ (576-dim vector)
       ▼
[Dropout (p=0.2)]
       │
       ▼
[Dense Classification Head (576 -> 3)]
       │
       ▼
[Softmax / Sigmoid Probabilities]
```

---

## 3. Dataset Curation & Augmentation Strategy

To train without proprietary datasets:
1. **Public Benchmarks**:
   - **Open-NSFW Dataset** / **Falcon NSFW Dataset** (~20,000 public labeled images).
   - **COCO / ImageNet subsets** (5,000 diverse neutral images: clothes, beaches, gym workouts, medical diagrams to prevent false positives on bare skin).
2. **Data Augmentation**:
   - Random Horizontal Flip ($p=0.5$).
   - Random Affine Rotation ($\pm 15^\circ$).
   - Color Jitter (Brightness $\pm 0.1$, Contrast $\pm 0.1$).
   - Random Erasing / Cutout ($p=0.2$) to force the network to recognize diverse body and texture cues rather than single local artifacts.

---

## 4. Quantization & Export Pipeline

1. **PyTorch Training**:
   - Loss function: Focal Loss ($\gamma=2.0, \alpha=[0.3, 0.4, 0.3]$) to combat class imbalance.
   - Optimizer: AdamW ($\text{lr}=10^{-4}$, weight decay $0.01$) with Cosine Annealing scheduler.
2. **ONNX Export**:
   - Export PyTorch `.pt` model to ONNX with dynamic batch sizing:
     ```python
     torch.onnx.export(
         model, dummy_input, "nsfw_mobilenet_v3.onnx",
         input_names=["input_tensor"], output_names=["class_probabilities"],
         dynamic_axes={"input_tensor": {0: "batch_size"}}
     )
     ```
3. **INT8 Post-Training Quantization (PTQ)**:
   - Use `onnxruntime.quantization.quantize_dynamic` (or static quantization with calibration data).
   - Results in a **3.8MB** model file with <0.5% accuracy loss.

---

## 5. Deployment Options

### Option A: Server-Side Inference (FastAPI)
- Uses `onnxruntime` in Python. Eliminates PyTorch from backend requirements (`pip install onnxruntime` vs `torch` saving >1.2GB disk space).
- Directly processes image frames extracted from posts and videos.

### Option B: Direct In-Browser Inference (Zero Backend Network Overhead)
- Ships `nsfw_mobilenet_v3.onnx` (3.8MB) inside the Chrome extension `extension/models/`.
- Executes via `onnxruntime-web` (WebAssembly / WebGPU):
  - Images and video canvas frames are scored directly on the user's machine in <10ms.
  - Zero image data ever leaves the user's browser, maximizing privacy.
