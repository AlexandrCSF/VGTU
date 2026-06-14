"""
Загрузка изображений Fashion-IQ по файлу img_urls.txt

Запуск:
    python download_images.py

Скачивает только те изображения, которые нужны для эксперимента
(упомянуты в captions/*.json и image_splits/*.json).
Уже скачанные пропускает.
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ─────────────────────────────────────────────────────────────
DATASET_DIR = Path("fashion-iq")
IMG_URLS_FILE = DATASET_DIR / "img_urls.txt"
IMAGES_DIR = DATASET_DIR / "images"
CATEGORIES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]
MAX_WORKERS = 8        # параллельных потоков скачивания
RETRY_COUNT = 3        # попыток при ошибке
TIMEOUT = 15           # секунд на соединение
# ─────────────────────────────────────────────────────────────

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print_lock = Lock()


def safe_print(*args):
    with print_lock:
        print(*args)


def load_url_map() -> dict[str, str]:
    """Читает img_urls.txt → {image_id: url}"""
    url_map = {}
    with open(IMG_URLS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                img_id = parts[0]
                url = parts[-1]
                url_map[img_id] = url
    return url_map


def collect_needed_ids() -> set[str]:
    """Собирает все image_id, упомянутые в аннотациях и splits."""
    needed = set()

    for cat in CATEGORIES:
        for split in SPLITS:
            # captions
            cap_path = DATASET_DIR / "captions" / f"cap.{cat}.{split}.json"
            if cap_path.exists():
                with open(cap_path, encoding="utf-8") as f:
                    for ann in json.load(f):
                        needed.add(ann["candidate"])
                        needed.add(ann.get("target", ""))

            # image_splits (gallery)
            spl_path = DATASET_DIR / "image_splits" / f"split.{cat}.{split}.json"
            if spl_path.exists():
                with open(spl_path, encoding="utf-8") as f:
                    needed.update(json.load(f))

    return needed


def download_one(img_id: str, url: str) -> tuple[str, str]:
    """Скачивает одно изображение. Возвращает (img_id, статус)."""
    out_path = IMAGES_DIR / f"{img_id}.jpg"
    if out_path.exists():
        return img_id, "skip"

    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return img_id, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return img_id, f"404"
            if attempt == RETRY_COUNT:
                return img_id, f"http_{e.code}"
            time.sleep(1)
        except Exception as e:
            if attempt == RETRY_COUNT:
                return img_id, f"err:{type(e).__name__}"
            time.sleep(1)

    return img_id, "failed"


def main():
    print("=" * 55)
    print("Загрузка изображений Fashion-IQ")
    print("=" * 55)

    print("\n[1/3] Читаем img_urls.txt...")
    if not IMG_URLS_FILE.exists():
        print(f"ОШИБКА: файл не найден: {IMG_URLS_FILE}")
        return
    url_map = load_url_map()
    print(f"  Всего URL: {len(url_map)}")

    print("\n[2/3] Определяем нужные изображения из аннотаций...")
    needed_ids = collect_needed_ids()
    print(f"  Нужно изображений: {len(needed_ids)}")

    # Ограничиваемся только теми, для которых есть URL
    to_download = {img_id: url_map[img_id] for img_id in needed_ids if img_id in url_map}
    missing_url = needed_ids - set(url_map.keys())

    already = sum(1 for img_id in to_download if (IMAGES_DIR / f"{img_id}.jpg").exists())
    print(f"  Уже скачано: {already}")
    print(f"  Нет URL в файле: {len(missing_url)}")
    print(f"  К скачиванию: {len(to_download) - already}")

    if not to_download:
        print("\nНечего скачивать.")
        return

    print(f"\n[3/3] Скачиваем ({MAX_WORKERS} потоков)...\n")

    stats = {"ok": 0, "skip": 0, "err": 0, "404": 0}
    total = len(to_download)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, img_id, url): img_id
                   for img_id, url in to_download.items()}
        for future in as_completed(futures):
            img_id, status = future.result()
            done += 1
            if status == "ok":
                stats["ok"] += 1
            elif status == "skip":
                stats["skip"] += 1
            elif status == "404":
                stats["404"] += 1
            else:
                stats["err"] += 1

            if done % 100 == 0 or done == total:
                safe_print(
                    f"  [{done}/{total}] ok={stats['ok']} "
                    f"skip={stats['skip']} 404={stats['404']} err={stats['err']}"
                )

    print("\n" + "=" * 55)
    print(f"Готово.")
    print(f"  Скачано:       {stats['ok']}")
    print(f"  Уже было:      {stats['skip']}")
    print(f"  Не найдено:    {stats['404']}")
    print(f"  Ошибки:        {stats['err']}")
    print(f"  Файлы в:       {IMAGES_DIR.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
