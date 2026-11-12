"""
=============================================================
batch_utils.py — Helpers for Batch Analysis
=============================================================
Deliberately has ZERO dependency on torch/torchvision/cv2. Everything
here is plain file/zip/csv handling, kept separate from app.py so it
can be unit tested without loading the ML models or GPU.
=============================================================
"""
import os
import io
import csv
import uuid
import zipfile

# Same allow-list as the single-image /analyze route in app.py
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Safety caps so a huge drag-drop or a malicious zip can't take the server down
MAX_BATCH_IMAGES     = 100
MAX_ZIP_UNCOMPRESSED = 300 * 1024 * 1024  # 300 MB


def is_allowed_image(filename):
    """True if filename has an extension we know how to analyze."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


def extract_images_from_zip(file_storage, max_files=MAX_BATCH_IMAGES):
    """
    Given a Werkzeug FileStorage wrapping a .zip upload, return a list of
    (filename, bytes) tuples for every valid image inside it.

    Skips directories, hidden/system files (.DS_Store, __MACOSX), and
    anything outside the extension allow-list. Raises ValueError on a
    corrupt zip or if it looks like a zip bomb (checked via the zip's own
    recorded uncompressed sizes, before any bytes are actually inflated).
    """
    data = file_storage.read()
    if not data:
        raise ValueError('Uploaded zip is empty.')
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError('That file is not a valid .zip archive.')

    results = []
    total_uncompressed = 0
    try:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            if not base or base.startswith('.') or '__MACOSX' in info.filename:
                continue
            if not is_allowed_image(base):
                continue

            total_uncompressed += info.file_size
            if total_uncompressed > MAX_ZIP_UNCOMPRESSED:
                raise ValueError('Zip contents too large (over 300MB uncompressed).')

            if len(results) >= max_files:
                break
            results.append((base, zf.read(info)))
    finally:
        zf.close()

    return results


def new_batch_id():
    return uuid.uuid4().hex[:12]


def new_image_id():
    return uuid.uuid4().hex[:8]


def results_to_csv(rows):
    """
    rows: list of dicts with keys filename, predicted, forged_prob, auth_prob, trust.
    Returns CSV text as a str. (The frontend also builds this client-side from
    the same JSON so the export button works without a round trip — this
    server-side version exists in case it's ever needed for a scripted export.)
    """
    buf = io.StringIO()
    fieldnames = ['filename', 'predicted', 'forged_prob', 'auth_prob', 'trust']
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


class BatchStore:
    """
    In-memory store mapping batch_id -> {image_id: record}.

    Good enough for this project's scope (single-process Flask dev server /
    small local deployment). NOT persisted across restarts — Flask's debug
    reloader restarting the process will wipe it, and it won't work across
    multiple gunicorn workers without moving to a shared store. That's a
    known, acceptable limitation here, not a bug.
    """
    def __init__(self):
        self._data = {}

    def create_batch(self):
        batch_id = new_batch_id()
        self._data[batch_id] = {}
        return batch_id

    def add_image(self, batch_id, image_id, record):
        self._data.setdefault(batch_id, {})[image_id] = record

    def get_image(self, batch_id, image_id):
        return self._data.get(batch_id, {}).get(image_id)

    def get_batch(self, batch_id):
        return self._data.get(batch_id, {})

    def batch_exists(self, batch_id):
        return batch_id in self._data
