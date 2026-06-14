import sys
import json
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

DATASET_DIR = Path("fashion-iq")
CATEGORY = "dress"
SPLIT = "val"
MAX_QUERIES = 500
MAX_GALLERY = None
K = 10
NOISE_FRACTION = 0.20
IMBALANCE_FRACTION = 0.30
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_clip():
    try:
        import clip as openai_clip
        model, preprocess = openai_clip.load("ViT-B/32", device=DEVICE)
        model.eval()
        return model, preprocess, openai_clip
    except ImportError:
        print("ОШИБКА: установите CLIP: pip install clip-by-openai")
        sys.exit(1)

def load_dataset() -> Tuple[List[Dict], List[str]]:
    ann_path = DATASET_DIR / "captions" / f"cap.{CATEGORY}.{SPLIT}.json"
    split_path = DATASET_DIR / "image_splits" / f"split.{CATEGORY}.{SPLIT}.json"

    if not ann_path.exists() or not split_path.exists():
        print(
            "\nFashion-IQ не найден. Инструкция:\n"
            "  git clone https://github.com/XiaoxiaoGuo/fashion-iq\n"
            "  Скопируйте images/, captions/, image_splits/ в ./fashion_iq/\n"
            f"  Ожидаемые файлы:\n  • {ann_path}\n  • {split_path}\n"
        )
        sys.exit(1)

    with open(ann_path, encoding="utf-8") as f:
        annotations = json.load(f)
    with open(split_path, encoding="utf-8") as f:
        gallery_ids = json.load(f)

    if MAX_QUERIES:
        annotations = annotations[:MAX_QUERIES]
    if MAX_GALLERY:
        gallery_ids = gallery_ids[:MAX_GALLERY]

    return annotations, gallery_ids


class FeatureExtractor:
    def __init__(self, model, preprocess, clip_lib):
        self.model = model
        self.preprocess = preprocess
        self.clip_lib = clip_lib
        self.dim = 512

    def _img_path(self, img_id: str) -> Path:
        return DATASET_DIR / "images" / f"{img_id}.jpg"

    @torch.no_grad()
    def encode_image(self, img_id: str) -> Optional[np.ndarray]:
        from PIL import Image
        path = self._img_path(img_id)
        if not path.exists():
            return None
        try:
            img = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            feat = self.model.encode_image(img).cpu().float().numpy()[0]
            return feat / (np.linalg.norm(feat) + 1e-8)
        except Exception:
            return None

    @torch.no_grad()
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        if not text.strip():
            return None
        try:
            tokens = self.clip_lib.tokenize([text], truncate=True).to(DEVICE)
            feat = self.model.encode_text(tokens).cpu().float().numpy()[0]
            return feat / (np.linalg.norm(feat) + 1e-8)
        except Exception:
            return None

    def encode_gallery(self, gallery_ids: List[str]) -> Dict[str, np.ndarray]:
        """Предвычисляет CLIP-признаки для всего gallery."""
        feats = {}
        for gid in tqdm(gallery_ids, desc="  Gallery embeddings", leave=False):
            f = self.encode_image(gid)
            if f is not None:
                feats[gid] = f
        return feats


# ─────────────────────────────────────────────────────────────
# Оценка качества модальностей Q_m ∈ [0, 1]
# ─────────────────────────────────────────────────────────────
def quality_text(text: str) -> float:
    """Качество текстовой модальности: длина + лексическое разнообразие."""
    if not text or not text.strip():
        return 0.0
    words = text.split()
    length_score = min(len(words) / 20.0, 1.0)
    ttr = len(set(w.lower() for w in words)) / max(len(words), 1)
    return 0.6 * length_score + 0.4 * ttr


def quality_image(img_id: str) -> float:
    """Качество изображения: резкость (Лапласиан) + контраст."""
    path = DATASET_DIR / "images" / f"{img_id}.jpg"
    if not path.exists():
        return 0.0
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    sharpness = min(cv2.Laplacian(img, cv2.CV_64F).var() / 500.0, 1.0)
    contrast = min(float(img.std()) / 64.0, 1.0)
    return 0.6 * sharpness + 0.4 * contrast


# ─────────────────────────────────────────────────────────────
# Fusion стратегии (скор релевантности для одной пары запрос–кандидат)
# ─────────────────────────────────────────────────────────────
def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # векторы уже нормированы


def early_fusion_score(
    img_feat: Optional[np.ndarray],
    txt_feat: Optional[np.ndarray],
    cand_feat: np.ndarray,
    dim: int = 512,
) -> float:
    """
    Ранний fusion (data-level): конкатенация признаков запроса.
    Запрос = concat(img, txt).
    Кандидат расширяется нулями до той же размерности.
    """
    q_img = img_feat if img_feat is not None else np.zeros(dim)
    q_txt = txt_feat if txt_feat is not None else np.zeros(dim)
    query = np.concatenate([q_img, q_txt])
    gallery = np.concatenate([cand_feat, np.zeros(dim)])
    norm = np.linalg.norm(query) * np.linalg.norm(gallery)
    return float(np.dot(query, gallery) / (norm + 1e-8))


def late_fusion_score(
    img_feat: Optional[np.ndarray],
    txt_feat: Optional[np.ndarray],
    cand_feat: np.ndarray,
    w_img: float = 0.5,
    w_text: float = 0.5,
) -> float:
    """
    Поздний fusion (decision-level): взвешенная сумма оценок по модальностям.
    r = w_img * cos(img, cand) + w_text * cos(txt, cand)
    """
    s_img = cos(img_feat, cand_feat) if img_feat is not None else 0.0
    s_txt = cos(txt_feat, cand_feat) if txt_feat is not None else 0.0
    w_i = w_img if img_feat is not None else 0.0
    w_t = w_text if txt_feat is not None else 0.0
    total_w = w_i + w_t
    if total_w < 1e-8:
        return 0.0
    return (w_i * s_img + w_t * s_txt) / total_w


def hybrid_fusion_score(
    img_feat: Optional[np.ndarray],
    txt_feat: Optional[np.ndarray],
    cand_feat: np.ndarray,
) -> float:
    """
    Гибридный fusion (model-level): взаимодействие в совместном CLIP-пространстве.
    Запрос = normalize(img + txt) — эмулирует cross-modal attention в общем пространстве CLIP.
    """
    parts = []
    if img_feat is not None:
        parts.append(img_feat)
    if txt_feat is not None:
        parts.append(txt_feat)
    if not parts:
        return 0.0
    combined = np.sum(parts, axis=0)
    norm = np.linalg.norm(combined)
    if norm < 1e-8:
        return 0.0
    combined = combined / norm
    return cos(combined, cand_feat)


def adaptive_fusion_score(
    img_feat: Optional[np.ndarray],
    txt_feat: Optional[np.ndarray],
    cand_feat: np.ndarray,
    q_img: float,
    q_txt: float,
) -> float:
    """
    Адаптивный fusion: r = Σ(Q_m · r_m) / Σ Q_m
    Веса вычисляются динамически на основе оценок качества модальностей.
    """
    terms = []
    weights = []

    if img_feat is not None and q_img > 0:
        terms.append(q_img * cos(img_feat, cand_feat))
        weights.append(q_img)

    if txt_feat is not None and q_txt > 0:
        terms.append(q_txt * cos(txt_feat, cand_feat))
        weights.append(q_txt)

    # Гибридный компонент (произведение весов отражает взаимодействие)
    if img_feat is not None and txt_feat is not None and q_img > 0 and q_txt > 0:
        q_cross = q_img * q_txt
        r_hybrid = hybrid_fusion_score(img_feat, txt_feat, cand_feat)
        terms.append(q_cross * r_hybrid)
        weights.append(q_cross)

    total_w = sum(weights)
    if total_w < 1e-8:
        return 0.0
    return sum(terms) / total_w


# ─────────────────────────────────────────────────────────────
# Применение шума к аннотациям
# ─────────────────────────────────────────────────────────────
def apply_noise(
    annotations: List[Dict],
    gallery_ids: List[str],
    noise_type: str,
    fraction: float,
) -> List[Dict]:
    result = [dict(a) for a in annotations]
    n_noisy = int(len(result) * fraction)
    indices = random.sample(range(len(result)), n_noisy)

    for i in indices:
        if noise_type == "text_noise":
            # Заменяем текст на нерелевантный
            result[i]["captions"] = ["random unrelated text", "some irrelevant description"]
        elif noise_type == "image_noise":
            # Подставляем случайное изображение вместо reference
            result[i]["candidate"] = random.choice(gallery_ids)
        elif noise_type == "missing_modality":
            # Убираем одну модальность (50/50)
            if random.random() < 0.5:
                result[i]["captions"] = ["", ""]       # нет текста
            else:
                result[i]["candidate"] = "__MISSING__" # нет изображения
        elif noise_type == "imbalance":
            # Убираем текстовую модальность
            result[i]["captions"] = ["", ""]

    return result


# ─────────────────────────────────────────────────────────────
# Recall@K
# ─────────────────────────────────────────────────────────────
def recall_at_k(
    annotations: List[Dict],
    gallery_feats: Dict[str, np.ndarray],
    extractor: FeatureExtractor,
    fusion: str,
    k: int = 10,
    with_quality: bool = False,
) -> float:
    hits = 0
    gallery_ids = list(gallery_feats.keys())

    for ann in annotations:
        ref_id = ann["candidate"]
        target_id = ann["target"]
        caption = " ".join(ann.get("captions", ["", ""])).strip()

        # Извлекаем признаки запроса
        img_feat = None if ref_id == "__MISSING__" else extractor.encode_image(ref_id)
        txt_feat = extractor.encode_text(caption) if caption else None

        # Оценки качества
        q_img = quality_image(ref_id) if (with_quality and ref_id != "__MISSING__") else 0.8
        q_txt = quality_text(caption) if with_quality else 0.8

        # Скоры по всем кандидатам
        scores = np.zeros(len(gallery_ids))
        for j, gid in enumerate(gallery_ids):
            cf = gallery_feats[gid]
            if fusion == "early":
                scores[j] = early_fusion_score(img_feat, txt_feat, cf, extractor.dim)
            elif fusion == "late":
                scores[j] = late_fusion_score(img_feat, txt_feat, cf)
            elif fusion == "hybrid":
                scores[j] = hybrid_fusion_score(img_feat, txt_feat, cf)
            elif fusion == "adaptive":
                scores[j] = adaptive_fusion_score(img_feat, txt_feat, cf, q_img, q_txt)

        top_k_ids = [gallery_ids[i] for i in np.argsort(scores)[::-1][:k]]
        if target_id in top_k_ids:
            hits += 1

    return hits / len(annotations) if annotations else 0.0


# ─────────────────────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Мультимодальный поиск: сравнение fusion-стратегий")
    print(f"Fashion-IQ | категория: {CATEGORY} | split: {SPLIT} | Recall@{K}")
    print("=" * 65)

    # Загрузка данных
    print("\n[1/4] Загрузка Fashion-IQ...")
    annotations, gallery_ids = load_dataset()
    print(f"  Запросов: {len(annotations)}, Gallery: {len(gallery_ids)}")

    # Загрузка CLIP
    print("\n[2/4] Загрузка CLIP ViT-B/32...")
    model, preprocess, clip_lib = load_clip()
    extractor = FeatureExtractor(model, preprocess, clip_lib)
    print(f"  Устройство: {DEVICE}")

    # Предвычисление gallery embeddings
    print("\n[3/4] Вычисление gallery embeddings...")
    gallery_feats = extractor.encode_gallery(gallery_ids)
    print(f"  Успешно закодировано: {len(gallery_feats)} / {len(gallery_ids)} изображений")

    # Условия данных
    conditions = [
        ("Чистые данные",              None,               None),
        ("Шум в тексте (20%)",         "text_noise",       NOISE_FRACTION),
        ("Шум в изображении (20%)",    "image_noise",      NOISE_FRACTION),
        ("Отсутствие модальности",     "missing_modality", NOISE_FRACTION),
        ("Дисбаланс модальностей (30%)","imbalance",       IMBALANCE_FRACTION),
    ]

    fusion_configs = [
        ("early",    "Ранний fusion",        False),
        ("late",     "Поздний fusion",       False),
        ("hybrid",   "Гибридный fusion",     False),
        ("adaptive", "Адаптивный (предл.)", True),
    ]

    # Оценка
    print("\n[4/4] Вычисление Recall@10...\n")
    results: Dict[str, Dict[str, float]] = {}

    for cond_name, noise_type, fraction in conditions:
        print(f"  Условие: {cond_name}")
        if noise_type:
            noisy_ann = apply_noise(annotations, gallery_ids, noise_type, fraction)
        else:
            noisy_ann = annotations

        results[cond_name] = {}
        for ftype, fname, with_q in fusion_configs:
            r = recall_at_k(
                noisy_ann, gallery_feats, extractor,
                fusion=ftype, k=K, with_quality=with_q,
            )
            results[cond_name][ftype] = r
            print(f"    {fname:25s} Recall@{K} = {r:.4f}")
        print()

    # Вывод итоговой таблицы
    print("=" * 75)
    print(f"ИТОГОВАЯ ТАБЛИЦА: Recall@{K}")
    print("=" * 75)

    col_w = 28
    headers = ["Условие данных", "Ранний", "Поздний", "Гибридный", "Адаптивный"]
    ftypes = ["early", "late", "hybrid", "adaptive"]

    try:
        from tabulate import tabulate
        rows = [
            [cname] + [f"{results[cname][ft]:.4f}" for ft in ftypes]
            for cname, *_ in conditions
        ]
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left"))
    except ImportError:
        sep = "+" + "+".join(["-" * (col_w + 2)] * len(headers)) + "+"
        header_row = "| " + " | ".join(f"{h:{col_w}s}" for h in headers) + " |"
        print(sep)
        print(header_row)
        print(sep)
        for cname, *_ in conditions:
            row = [cname] + [f"{results[cname][ft]:.4f}" for ft in ftypes]
            print("| " + " | ".join(f"{str(v):{col_w}s}" for v in row) + " |")
        print(sep)

    # Сравнение с данными статьи
    paper = {
        "Чистые данные":               [0.71, 0.67, 0.79, 0.82],
        "Шум в тексте (20%)":          [0.53, 0.63, 0.71, 0.76],
        "Шум в изображении (20%)":     [0.58, 0.65, 0.74, 0.78],
        "Отсутствие модальности":      [0.34, 0.64, 0.68, 0.73],
        "Дисбаланс модальностей (30%)":[0.49, 0.62, 0.70, 0.75],
    }

    print("\n" + "=" * 75)
    print("ДАННЫЕ ИЗ СТАТЬИ (таблица 1)")
    print("=" * 75)
    try:
        from tabulate import tabulate
        rows = [
            [cname] + [f"{v:.2f}" for v in vals]
            for cname, vals in paper.items()
        ]
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left"))
    except ImportError:
        for cname, vals in paper.items():
            print(f"  {cname}: " + ", ".join(f"{v:.2f}" for v in vals))

    # Сохраняем результаты в JSON
    output = {
        "config": {
            "category": CATEGORY, "split": SPLIT, "k": K,
            "max_queries": MAX_QUERIES, "seed": SEED,
            "noise_fraction": NOISE_FRACTION, "imbalance_fraction": IMBALANCE_FRACTION,
            "device": DEVICE,
        },
        "results": {
            cname: {ft: round(results[cname][ft], 4) for ft in ftypes}
            for cname, *_ in conditions
        },
        "paper_reference": paper,
    }
    out_path = Path("fusion_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {out_path.resolve()}")


if __name__ == "__main__":
    main()
