"""
=============================================================
EXIF Metadata Feature Extractor — Phase 2
=============================================================
Extracts a fixed-length numeric feature vector from an image's
EXIF metadata, designed to work even when EXIF is missing
(which is the common case for social-media-shared images).

Use this same module in:
  1. Dataset preprocessing (extract features for all training images)
  2. Flask inference (extract features for uploaded/captured images)

This guarantees train/inference consistency.
=============================================================
"""

from PIL import Image, ExifTags
import numpy as np

# Reverse-lookup EXIF tag names -> ids (PIL gives ids, we want names)
_TAG_NAME_TO_ID = {v: k for k, v in ExifTags.TAGS.items()}

# Known editing / social-media software strings.
# Presence of these in the "Software" EXIF tag is a moderate forgery signal —
# NOT proof of forgery, but a useful correlated feature.
_EDITOR_KEYWORDS = [
    'photoshop', 'gimp', 'lightroom', 'picsart', 'snapseed',
    'facetune', 'canva', 'affinity', 'paint.net', 'pixlr',
    'illustrator', 'capture one', 'luminar'
]

EXIF_FEATURE_NAMES = [
    'has_exif',              # 0/1 — any EXIF block present at all
    'num_exif_tags_norm',    # tag count / 40 (normalized, capped at 1.0)
    'has_camera_make',       # 0/1
    'has_camera_model',      # 0/1
    'has_software_tag',      # 0/1 — any Software tag present
    'software_is_editor',    # 0/1 — Software tag matches known editing tool
    'has_gps',                # 0/1
    'has_datetime_original',  # 0/1
    'has_datetime_modified',  # 0/1
    'datetime_mismatch',      # 0/1 — Original vs Modified differ
    'has_thumbnail',          # 0/1
    'has_exif_version',       # 0/1
    'has_orientation',        # 0/1
    'has_resolution_info',    # 0/1 — XResolution/YResolution present
    'has_flash_info',         # 0/1
    'has_lens_info',          # 0/1
]
EXIF_FEATURE_DIM = len(EXIF_FEATURE_NAMES)


def extract_exif_features(pil_img_or_path):
    """
    Extract a fixed-length (EXIF_FEATURE_DIM,) float32 numpy vector
    from a PIL Image or a file path.

    Never raises — returns an all-"no EXIF" vector on any failure,
    so this is always safe to call in a training loop or Flask route.
    """
    try:
        if isinstance(pil_img_or_path, str):
            img = Image.open(pil_img_or_path)
        else:
            img = pil_img_or_path

        exif = img.getexif()
        # Also merge in the IFD extension blocks (GPS, Exif sub-IFD)
        # which getexif() alone doesn't always expose on older Pillow.
        tags = {}
        if exif:
            for tag_id, value in exif.items():
                name = ExifTags.TAGS.get(tag_id, tag_id)
                tags[name] = value
            try:
                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                for tag_id, value in exif_ifd.items():
                    name = ExifTags.TAGS.get(tag_id, tag_id)
                    tags[name] = value
            except Exception:
                pass
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                if gps_ifd:
                    tags['GPSInfo'] = gps_ifd
            except Exception:
                pass

        has_exif = 1.0 if len(tags) > 0 else 0.0
        num_tags_norm = min(len(tags) / 40.0, 1.0)

        has_make = 1.0 if 'Make' in tags else 0.0
        has_model = 1.0 if 'Model' in tags else 0.0

        software = str(tags.get('Software', '')).lower()
        has_software = 1.0 if software.strip() else 0.0
        is_editor = 1.0 if any(k in software for k in _EDITOR_KEYWORDS) else 0.0

        has_gps = 1.0 if 'GPSInfo' in tags and tags['GPSInfo'] else 0.0

        has_dt_orig = 1.0 if 'DateTimeOriginal' in tags else 0.0
        has_dt_mod = 1.0 if 'DateTime' in tags else 0.0
        dt_mismatch = 0.0
        if has_dt_orig and has_dt_mod:
            dt_mismatch = 1.0 if tags['DateTimeOriginal'] != tags['DateTime'] else 0.0

        # Thumbnail check: look for a second embedded JPEG SOI marker within
        # the raw EXIF blob (the 1st-IFD thumbnail is itself a small JPEG).
        raw_exif = img.info.get('exif', b'')
        has_thumb = 1.0 if (raw_exif and raw_exif.count(b'\xff\xd8\xff') > 0) else 0.0

        has_exif_version = 1.0 if 'ExifVersion' in tags else 0.0
        has_orientation = 1.0 if 'Orientation' in tags else 0.0
        has_resolution = 1.0 if ('XResolution' in tags and 'YResolution' in tags) else 0.0
        has_flash = 1.0 if 'Flash' in tags else 0.0
        has_lens = 1.0 if ('LensModel' in tags or 'LensMake' in tags) else 0.0

        vec = np.array([
            has_exif, num_tags_norm, has_make, has_model,
            has_software, is_editor, has_gps, has_dt_orig,
            has_dt_mod, dt_mismatch, has_thumb, has_exif_version,
            has_orientation, has_resolution, has_flash, has_lens
        ], dtype=np.float32)
        return vec

    except Exception:
        # Any failure (corrupt EXIF, unsupported format, etc.) -> "no EXIF" vector
        return np.zeros(EXIF_FEATURE_DIM, dtype=np.float32)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        vec = extract_exif_features(sys.argv[1])
        for name, val in zip(EXIF_FEATURE_NAMES, vec):
            print(f'{name:25s}: {val}')
    else:
        print('Usage: python exif_utils.py <image_path>')
