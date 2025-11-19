"""
    Function: CIFAR-10 CLIP ⟂ Milvus ⟂ VLM (RAG)

    Run a full CIFAR-10 k-shot experiment using:
        • CLIP image embeddings  
        • Milvus as a vector database for support-set retrieval  
        • (Optional) an Ollama-served VLM with CLIP-RAG neighbor hints  
        • (Optional) a ResNet-18 CNN baseline

    For each k in `k_shots_list`, the function:
        1. Builds per-class CLIP support sets and stores them in a fresh Milvus collection.
        2. Evaluates VectorDB-only classification accuracy on a fixed 5k validation split.
        3. If enabled, evaluates VLM accuracy with non-binding CLIP-RAG suggestions.
        4. Optionally trains + evaluates a lightweight CNN baseline.
        5. Produces a Matplotlib figure showing accuracy vs. k.

    Returns
    -------
    results : dict
        Contains:
            - seed
            - llm_model
            - k_shots_list
            - vec_clip_curve
            - vlm_clip_rag_curve (or None)
            - vlm_only_acc
            - cnn_acc (or None)
    fig : matplotlib.figure.Figure
        Accuracy plot.

    Example
    -------
    >>> results, fig = run_cifar10_vlm_rag_experiment(
    ...     seed=123,
    ...     llm_model="llava:13b",
    ...     k_shots_list=(1, 2, 4, 8, 16),
    ...     topk_retrieve=3,
    ...     include_llm=True,
    ...     include_cnn_base=False,
    ...     ollama_url="http://localhost:11434",
    ...     milvus_host="127.0.0.1",
    ...     milvus_port="19530"
    ... )
    >>> print(results["vec_clip_curve"])
"""

def run_cifar10_vlm_rag_experiment(
    seed: int = 78,
    llm_model: str = "llava:13b",
    k_shots_list = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    *,
    topk_retrieve: int = 3,
    topk_vote: int = 1,
    vlm_eval_limit: int | None = 100,
    include_llm: bool = True,
    include_cnn_base: bool = False,
    # CNN train params
    batch_size_feats: int = 256,
    batch_size_train: int = 256,
    epochs_cnn: int = 4,
    # Infra defaults
    ollama_url: str = "http://localhost:11434",
    milvus_host: str = "127.0.0.1",
    milvus_port: str = "19530",
    milvus_db: str = "default"
):
    """
    Runs the exact mechanism of your original program with parameterized seed, LLM model, and shot list.
    Returns a dict of results and the Matplotlib figure object.
    """
    # ── Imports (same as your script) ────────────────────────────────────────────
    import os, random, time, json, io, base64, requests
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, Subset, random_split
    import open_clip
    from PIL import Image

    from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType,
        Collection, utility
    )

    # ── Repro (seed) ────────────────────────────────────────────────────────────
    SEED = int(seed)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # ── Devices (unchanged) ────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    clip_device = device
    print("Device:", device)

    # ── Experiment knobs (from params; same defaults) ──────────────────────────
    K_SHOTS_LIST        = list(k_shots_list)
    SUPPORT_POOL_PER_C  = max(K_SHOTS_LIST)
    TOPK_RETRIEVE       = int(topk_retrieve)
    TOPK_VOTE           = int(topk_vote)
    VLM_EVAL_LIMIT      = vlm_eval_limit
    INCLUDE_LLM         = bool(include_llm)
    INCLUDE_CNN_BASE    = bool(include_cnn_base)

    # CNN params (unchanged)
    BATCH_SIZE_FEATS = int(batch_size_feats)
    BATCH_SIZE_TRAIN = int(batch_size_train)
    EPOCHS_CNN       = int(epochs_cnn)

    # Milvus config (unchanged)
    MILVUS_ALIAS = "default"
    MILVUS_HOST  = milvus_host
    MILVUS_PORT  = milvus_port
    MILVUS_DB    = milvus_db

    # VLM (Ollama)
    VLM_MODEL  = llm_model
    OLLAMA_URL = ollama_url

    # CIFAR-10 class names (unchanged)
    CIFAR10_CLASSES = [
        "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
    ]

    # ────────────────────── 2) Data & CLIP model (unchanged) ───────────────────
    transform_eval = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    root = "./data"
    train_set_full = datasets.CIFAR10(root=root, train=True,  download=True, transform=transform_train)
    test_set       = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_eval)

    # Fixed VAL subset (unchanged)
    VAL_SIZE    = 5000
    val_indices = list(range(VAL_SIZE))
    val_set     = Subset(test_set, val_indices)
    val_loader  = DataLoader(val_set, batch_size=BATCH_SIZE_FEATS, shuffle=False, num_workers=2)

    # Raw (PIL) access for VLM
    raw_test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())

    # CLIP (ViT-B/32)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=clip_device
    )
    clip_model.eval()

    # CLIP text embeddings
    clip_prompts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    with torch.no_grad():
        text_tokens = open_clip.tokenize(clip_prompts).to(clip_device)
        clip_text_emb = clip_model.encode_text(text_tokens)
        clip_text_emb = clip_text_emb / clip_text_emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def extract_clip_feats(dataset, batch_size=256):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        feats, labels, idxs = [], [], []
        i0 = 0
        for x, y in tqdm(loader, desc="CLIP: extracting image embeddings"):
            x = x.to(clip_device)
            f = clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.detach().cpu().numpy())
            labels.append(y.numpy())
            bsz = y.shape[0]
            idxs.append(np.arange(i0, i0+bsz))
            i0 += bsz
        feats  = np.concatenate(feats, axis=0).astype(np.float32)
        labels = np.concatenate(labels, axis=0)
        idxs   = np.concatenate(idxs, axis=0)
        return feats, labels, idxs

    # Datasets aligned to VAL split
    clip_train_ds  = datasets.CIFAR10(root=root, train=True,  download=True, transform=clip_preprocess)
    clip_test_ds   = datasets.CIFAR10(root=root, train=False, download=True, transform=clip_preprocess)
    clip_val_ds    = Subset(clip_test_ds, val_indices)

    clip_train_feats, clip_train_labels, clip_train_idxs = extract_clip_feats(clip_train_ds, batch_size=BATCH_SIZE_FEATS)
    clip_val_feats,   clip_val_labels,   clip_val_idxs   = extract_clip_feats(clip_val_ds,   batch_size=BATCH_SIZE_FEATS)
    CLIP_EMBED_DIM = clip_train_feats.shape[1]
    print("CLIP train:", clip_train_feats.shape, "| CLIP val:", clip_val_feats.shape)

    # Build per-class shuffled pools for supports (unchanged)
    rng = np.random.default_rng(SEED)
    clip_class_to_indices = {c: np.where(clip_train_labels == c)[0] for c in range(10)}
    for c in range(10):
        rng.shuffle(clip_class_to_indices[c])
        clip_class_to_indices[c] = clip_class_to_indices[c][:SUPPORT_POOL_PER_C]

    print("Support pools per class:")
    for c in range(10):
        print(f" Class {c} ({CIFAR10_CLASSES[c]}): {len(clip_class_to_indices[c])} samples")

    # ────────────────────── 3) Milvus helpers (unchanged) ──────────────────────
    def connect_milvus():
        if not connections.has_connection(MILVUS_ALIAS):
            connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
        utility.list_collections()
        print("[Milvus] Connected.")

    def ensure_new_collection(name: str, dim: int) -> Collection:
        if utility.has_collection(name):
            utility.drop_collection(name)
        fields = [
            FieldSchema(name="pk",        dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="label",     dtype=DataType.INT64),
            FieldSchema(name="img_idx",   dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description=f"CIFAR10 CLIP support {name}")
        col = Collection(name=name, schema=schema)
        index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
        col.create_index(field_name="embedding", index_params=index_params)
        col.load()
        return col

    def build_clip_collection_for_k(k: int) -> Collection:
        name = f"cifar10_clip_k{k}"
        col  = ensure_new_collection(name, dim=CLIP_EMBED_DIM)

        sup_indices, sup_labels = [], []
        for c in range(10):
            chosen = clip_class_to_indices[c][:k]
            if len(chosen) < k:
                raise ValueError(f"Class {c} has only {len(chosen)} candidates; need {k}.")
            sup_indices.extend(chosen)
            sup_labels.extend([c]*k)

        sup_indices = np.array(sup_indices, dtype=int)
        sup_labels  = np.array(sup_labels,  dtype=int)
        sup_embeds  = clip_train_feats[sup_indices]    # already L2-normalized

        col.insert([sup_labels.astype(np.int64),
                    clip_train_idxs[sup_indices].astype(np.int64),
                    sup_embeds.astype(np.float32)])
        col.flush(); col.load()
        return col

    def milvus_search(col: Collection, queries: np.ndarray, topk: int):
        params = {"metric_type": "IP", "params": {"nprobe": 32}}
        return col.search(
            data=queries.astype(np.float32),
            anns_field="embedding",
            param=params,
            limit=topk,
            output_fields=["label","img_idx"]
        )

    # ────────────────────── 4) Predictors (unchanged) ──────────────────────────
    def predict_vector_only(col: Collection, q_feats: np.ndarray, topk_vote:int=1) -> np.ndarray:
        results = milvus_search(col, q_feats, topk=max(topk_vote,1))
        preds   = []
        for hits in results:
            if topk_vote == 1:
                preds.append(int(hits[0].entity.get("label")))
            else:
                labs   = [int(h.entity.get("label")) for h in hits]
                counts = np.bincount(labs, minlength=10)
                preds.append(labs[0] if counts.max()==1 else int(np.argmax(counts)))
        return np.array(preds, dtype=int)

    def _pil_to_b64(img: Image.Image, quality=90) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _vlm_prompt(neighbor_suggestions: str | None = None) -> str:
        base = (
            "Classify this CIFAR-10 image into exactly one of these classes:\n"
            f"{', '.join(CIFAR10_CLASSES)}.\n"
            "Answer with only the class name."
        )
        if neighbor_suggestions:
            base += (
                "\n\nNon-binding nearest-neighbor hints:\n"
                f"{neighbor_suggestions}\n"
                "Use them only if the image is consistent."
            )
        return base

    def _parse_pred_from_text(txt: str) -> str:
        low = (txt or "").strip().lower()
        for name in CIFAR10_CLASSES:
            if name in low:
                return name
        return low.split()[0] if low else "unknown"

    def _neighbor_hint_text(hits) -> str:
        lines = []
        for i, h in enumerate(hits, 1):
            lab = int(h.entity.get("label"))
            sim = float(h.distance)
            lines.append(f"{i}. class={CIFAR10_CLASSES[lab]}, similarity={sim:.3f}")
        return "\n".join(lines)

    def vlm_classify_image(img_pil: Image.Image, model=VLM_MODEL, neighbor_suggestions: str | None = None,
                           max_tokens=16, temperature=0.0) -> str:
        # prefer native /api/generate with "images"; fallback to /v1/chat/completions
        b64    = _pil_to_b64(img_pil)
        prompt = _vlm_prompt(neighbor_suggestions)

        gen_url = OLLAMA_URL.rstrip("/") + "/api/generate"
        payload_gen = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature}
        }
        try:
            r = requests.post(gen_url, json=payload_gen, timeout=120)
            if r.status_code != 404:
                r.raise_for_status()
                return r.json().get("response","").strip()
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise

        # fallback (OpenAI-style)
        data_url = "data:image/jpeg;base64," + b64
        chat_url = OLLAMA_URL.rstrip("/") + "/v1/chat/completions"
        payload_chat = {
            "model": model,
            "messages": [
                {"role":"user","content":[
                    {"type":"text","text": prompt},
                    {"type":"image_url","image_url":{"url": data_url}}
                ]}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        rc = requests.post(chat_url, json=payload_chat, timeout=120)
        rc.raise_for_status()
        j = rc.json()
        return j["choices"][0]["message"]["content"].strip()

    @torch.no_grad()
    def eval_vlm_with_clip_rag(col: Collection, include_rag: bool, limit: int | None = VLM_EVAL_LIMIT) -> float:
        idxs = val_indices if limit is None else val_indices[:limit]
        correct, total = 0, 0
        for j in tqdm(range(len(idxs)), desc=f"VLM eval (RAG={include_rag})"):
            i  = idxs[j]
            gt = test_set.targets[i]
            img_pil = transforms.ToPILImage()(raw_test_set[i][0])

            suggestions = None
            if include_rag:
                q_feat = clip_val_feats[j:j+1]  # j aligns with clip_val_ds order
                hits   = milvus_search(col, q_feat, topk=TOPK_RETRIEVE)[0]
                suggestions = _neighbor_hint_text(hits)

            txt  = vlm_classify_image(img_pil, model=VLM_MODEL, neighbor_suggestions=suggestions)
            name = _parse_pred_from_text(txt)
            pred = CIFAR10_CLASSES.index(name) if name in CIFAR10_CLASSES else -1
            if pred == gt:
                correct += 1
            total += 1
        return correct / total

    # ────────────────────── 5) Optional CNN baseline (unchanged) ───────────────
    def run_cnn_baseline() -> float:
        train_len = 48000
        val_len   = len(train_set_full) - train_len
        cnn_train, _ = random_split(train_set_full, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(SEED))
        cnn_loader = DataLoader(cnn_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2)

        cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 10)
        cnn_model = cnn_model.to(device)

        optim = torch.optim.AdamW(cnn_model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS_CNN)
        criterion = nn.CrossEntropyLoss()

        def train_cnn(model, loader, epochs):
            model.train()
            for ep in range(1, epochs+1):
                losses, correct, total = [], 0, 0
                for x,y in tqdm(loader, desc=f"CNN Train {ep}/{epochs}"):
                    x, y = x.to(device), y.to(device)
                    optim.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optim.step()
                    losses.append(loss.item())
                    pred = out.argmax(1)
                    correct += (pred==y).sum().item()
                    total   += y.numel()
                sched.step()
                print(f"Epoch {ep}: loss={np.mean(losses):.4f}, acc={(correct/total)*100:.2f}%")

        @torch.no_grad()
        def eval_cnn(model, dataloader):
            model.eval()
            preds, labs = [], []
            for x,y in tqdm(dataloader, desc="CNN Eval"):
                x = x.to(device)
                out = model(x)
                preds.append(out.argmax(1).detach().cpu().numpy())
                labs.append(y.numpy())
            preds = np.concatenate(preds); labs = np.concatenate(labs)
            return accuracy_score(labs, preds)

        train_cnn(cnn_model, cnn_loader, EPOCHS_CNN)
        val_acc = eval_cnn(cnn_model, val_loader)
        print(f"[CNN baseline] accuracy on VAL({len(val_set)}): {val_acc*100:.2f}%")
        return val_acc

    cnn_acc = run_cnn_baseline() if INCLUDE_CNN_BASE else None

    # ────────────────────── 6) Main: build supports, evaluate, plot (unchanged) ─
    connect_milvus()

    # VLM-only (no RAG) once
    tmp_col       = build_clip_collection_for_k(max(1, K_SHOTS_LIST[0]))
    vlm_only_acc  = eval_vlm_with_clip_rag(tmp_col, include_rag=False)
    print(f"[VLM-only baseline] {vlm_only_acc*100:.2f}%")

    vec_clip_curve     = []
    vlm_clip_rag_curve = []

    print("SHOT LIST:", K_SHOTS_LIST)

    for k in K_SHOTS_LIST:
        print(f"\n=== k={k} shots per class ===")
        col = build_clip_collection_for_k(k)

        # VecDB-only
        preds_vec = predict_vector_only(col, clip_val_feats, topk_vote=TOPK_VOTE)
        acc_vec   = accuracy_score(clip_val_labels, preds_vec)
        vec_clip_curve.append(acc_vec)
        print(f"[CLIP VecDB-only] k={k}: {acc_vec*100:.2f}%")

        # VLM + RAG (non-binding)
        if INCLUDE_LLM:
            acc_vlm = eval_vlm_with_clip_rag(col, include_rag=True)
            vlm_clip_rag_curve.append(acc_vlm)
            print(f"[VLM + CLIP-RAG] k={k}: {acc_vlm*100:.2f}%")

    # Plot with uniform x spacing but original labels
    x = np.arange(len(K_SHOTS_LIST))
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()

    ax.plot(x, np.array(vec_clip_curve)*100, marker='o',
            label='VectorDB-only (CLIP embeddings, k-shots)')

    if INCLUDE_LLM:
        ax.plot(x, np.array(vlm_clip_rag_curve)*100, marker='s',
                label='VLM + CLIP-RAG (non-binding, k-shots)')

    ax.axhline(vlm_only_acc*100.0, linestyle='-',  label='VLM-only')
    if cnn_acc is not None:
        ax.axhline(cnn_acc*100.0, linestyle='--', label='CNN baseline')

    ax.set_title(f"CIFAR-10 with CLIP Embeddings: k-shot VecDB & VLM+RAG vs Baselines (SEED={SEED})")
    ax.set_xlabel("Shots per class (k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(K_SHOTS_LIST)
    ax.grid(True, linestyle=':')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # ── Return useful artifacts ────────────────────────────────────────────────
    results = {
        "seed": SEED,
        "llm_model": VLM_MODEL,
        "k_shots_list": K_SHOTS_LIST,
        "vec_clip_curve": vec_clip_curve,
        "vlm_clip_rag_curve": vlm_clip_rag_curve if INCLUDE_LLM else None,
        "vlm_only_acc": vlm_only_acc,
        "cnn_acc": cnn_acc
    }
    return results, fig
