# CIFAR-10: CLIP ⟂ Milvus ⟂ VLM (RAG) — Few-Shot Evaluation Pipeline

This repository implements a full **k-shot CIFAR-10** experiment combining:

* **CLIP (ViT-B/32)** image embeddings
* **Milvus** vector database for support-set retrieval
* **Ollama-served Vision-Language Models (VLMs)** with CLIP-RAG hints
* **Optional CNN baseline** (ResNet-18)
* **Automatic sweeping over k-shots** and accuracy plotting

The goal is to evaluate how **few-shot support size** affects:

* CLIP-based VectorDB retrieval accuracy
* VLM accuracy with non-binding RAG hints
* Baseline VLM-only accuracy
* (Optional) CNN baseline accuracy

The main entrypoint is:

```
run_cifar10_vlm_rag_experiment(...)
```

which orchestrates the full workflow: loading CIFAR-10, embedding with CLIP, building Milvus collections, running inference with VecDB / VLM / CNN, and plotting the accuracy curves.

---

## 🔧 1. Requirements

### **Hardware**

* macOS or Linux recommended
* GPU optional

  * On macOS, **MPS** will be used automatically
  * On Linux, CPU is fine but slower

### **Software prerequisites**

Before running, ensure you have:

#### **Python 3.10+**

Create a fresh environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### **Ollama (required for VLM)**

Install from: [https://ollama.com/download](https://ollama.com/download)

Pull the model you intend to use:

```bash
ollama pull llava:13b
```

Or any other VLM compatible with `/api/generate`.

Ensure Ollama server is running:

```bash
ollama serve
```

Default URL your script expects:

```
http://localhost:11434
```

---

#### **Milvus (required for VectorDB)**

You can run standalone Milvus using Docker Compose:

```bash
curl -s https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker-compose/docker-compose.yml -o docker-compose.yml
docker compose up -d
```

Then confirm it’s alive:

```
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
```

---

#### **Python dependencies**

Install all required packages:

```bash
pip install \
    torch torchvision torchaudio \
    numpy scipy scikit-learn \
    matplotlib tqdm \
    requests pillow \
    pymilvus==2.4.4 \
    open-clip-torch \
    ipykernel
```

macOS (MPS) users may optionally install nightly PyTorch builds.

---

## 📦 2. Directory Structure

Example suggested structure:

```
project/
│
├── main.py               # where run_cifar10_vlm_rag_experiment lives
├── README.md
└── data/                 # CIFAR-10 auto-download location
```

No manual dataset download needed.

---

## 🚀 3. Running the Experiment

Inside Python:

```python
from main import run_cifar10_vlm_rag_experiment

results, fig = run_cifar10_vlm_rag_experiment(
    seed=123,
    llm_model="llava:13b",
    k_shots_list=(1, 2, 4, 8, 16, 32),
    topk_retrieve=3,
    topk_vote=1,
    include_llm=True,
    include_cnn_base=False,
    ollama_url="http://localhost:11434",
    milvus_host="127.0.0.1",
    milvus_port="19530",
)
```

The function prints real-time progress for:

* CLIP embedding extraction
* Milvus collection creation & indexing
* VectorDB-only evaluation
* VLM-only evaluation
* VLM+RAG evaluation
* Optional CNN baseline

And finally returns:

* `results` — dict of all curves and accuracies
* `fig` — Matplotlib figure handle

---

## 📊 4. Output

You will see a plot like:

* VectorDB-only accuracy vs k
* VLM + RAG accuracy vs k
* VLM-only baseline (horizontal line)
* Optional CNN baseline (horizontal dashed line)

The `results` dictionary looks like:

```python
{
    "seed": 123,
    "llm_model": "llava:13b",
    "k_shots_list": [...],
    "vec_clip_curve": [...],
    "vlm_clip_rag_curve": [...],
    "vlm_only_acc": 0.xxx,
    "cnn_acc": None or float
}
```

---

## 🧱 5. Implementation Notes

* Support sets are formed using **nested k-shot pools** per class
* CLIP features are **L2-normalized** before insertion into Milvus
* Similarities use **Inner Product (IP)**
* RAG hints are **non-binding** textual summaries of nearest neighbors
* Evaluation uses a **fixed 5,000-image validation split**
* Collections for each k are created/dropped automatically

---

## 🧪 6. Reproducibility

The function sets:

* Python `random` seed
* NumPy seed
* PyTorch seed

so results are deterministic for a given `seed`.

---

## 🧩 7. Troubleshooting

### **Milvus connection error**

* Ensure Docker Milvus is running
* Verify port 19530 is open
* Try:

  ```python
  from pymilvus import connections
  connections.connect(host="127.0.0.1", port="19530")
  ```

### **Ollama model not found**

Run:

```
ollama pull llava:13b
```

Make sure `ollama serve` is running.

### **Slow CLIP extraction**

* Consider lowering batch size
* Enable GPU
* Use smaller model like `"ViT-B-16"`
