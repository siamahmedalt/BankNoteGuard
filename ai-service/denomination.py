"""Conservative OCR fallback for denomination, independent of authenticity.
Only use after the trained classifier accepts a note. No color-based guessing.
"""
import csv
import io
import logging
import re
import subprocess
import unicodedata
from PIL import Image, ImageOps, ImageFilter

LOG = logging.getLogger('banknoteguard.denomination')
DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
WORDS = {'পাঁচশত': 500, 'পাঁচশ': 500, 'পাচশত': 500, 'একহাজার': 1000,
         'fivehundred': 500, 'onethousand': 1000}


def parse_evidence(tsv):
    """Match complete amounts, never substrings of serial numbers; abstain on conflict."""
    lines = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t'):
        try:
            if row['level'] != '5' or float(row['conf']) < 65:
                continue
            token = unicodedata.normalize('NFC', row['text']).strip().lower()
            token = token.translate(DIGITS)
            key = tuple(row[k] for k in ('page_num','block_num','par_num','line_num'))
            lines.setdefault(key, []).append((token, float(row['conf'])))
        except (KeyError, TypeError, ValueError):
            continue
    evidence = set()
    for words in lines.values():
        for token, conf in words:
            # Strict complete-token match; allow a printed currency sign.
            m = re.fullmatch(r'(?:৳|tk\.?)?\s*(500|1000|1,000)(?:৳|tk\.?)?', token)
            if m and conf >= 78:
                evidence.add(int(m.group(1).replace(',', '')))
        for i in range(len(words)):
            for n in (1, 2, 3):
                chunk = words[i:i+n]
                if len(chunk) != n: continue
                text = ''.join(t for t,c in chunk).replace('টাকা','').replace('taka','')
                if text in WORDS and min(c for t,c in chunk) >= 65:
                    evidence.add(WORDS[text])
    return next(iter(evidence)) if len(evidence) == 1 else None


def _panels(image):
    """Magnify note corners and central text without OCR of serial-number substrings."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    if image.height > image.width * 1.25:
        image = image.rotate(90, expand=True)
    w,h = image.size
    regions = [(0,0,1,1), (0,0,.42,.37), (.58,0,1,.37),
               (0,.63,.42,1), (.58,.63,1,1), (.25,.13,.75,.60)]
    for i,(a,b,c,d) in enumerate(regions):
        crop = image.crop((int(a*w),int(b*h),max(int(c*w),1),max(int(d*h),1)))
        if min(crop.size) < 2: continue
        grey = ImageOps.autocontrast(crop.convert('L'), cutoff=1)
        target = 1050 if i == 0 else 650
        scale = min(target / grey.width, 350 / grey.height)
        grey = grey.resize((max(1,round(grey.width*scale)),max(1,round(grey.height*scale))), Image.Resampling.LANCZOS)
        yield grey.filter(ImageFilter.UnsharpMask(radius=1.3, percent=160, threshold=3))


def recognize_denomination(image, full_image=None):
    """One bounded local OCR process per scan; abstain if unavailable or uncertain."""
    images = [image] if full_image is None or full_image is image else [image, full_image]
    panels = [p for im in images for p in _panels(im)]
    if not panels: return None
    # Single image with white gutters; preserves all candidate text in one OCR call.
    canvas = Image.new('L',(1120, sum(p.height+36 for p in panels)+36),255)
    y=18
    for p in panels:
        canvas.paste(p,((canvas.width-p.width)//2,y)); y += p.height+36
    data=io.BytesIO(); canvas.save(data,format='PNG')
    try:
        result = subprocess.run(['tesseract','stdin','stdout','-l','eng+ben','--psm','11','tsv'],
                                input=data.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                timeout=18,check=True)
        return parse_evidence(result.stdout.decode('utf-8',errors='replace'))
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning('Denomination OCR unavailable: %s',exc)
        return None
