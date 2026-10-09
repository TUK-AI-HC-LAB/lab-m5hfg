"""Create a traceable input/GT/official-heatmap preview from this run."""
from pathlib import Path
import json
from PIL import Image, ImageDraw
root = Path('/home/test/data/mvtec/bottle')
run = Path('/home/test/musc_results/bottle_paper_20261002')
out = Path('/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method10/source/results/bottle_paper_20261002')
types = ['broken_large','broken_small','contamination']
canvas = Image.new('RGB',(900,960),'white')
draw = ImageDraw.Draw(canvas)
for col,title in enumerate(['Input','Ground truth','MuSc heatmap (per-image normalized)']):
    draw.text((col*300+5,5),title,fill='black')
manifest=[]
for row,kind in enumerate(types):
    path=sorted((root/'test'/kind).glob('*.png'))[0]
    mask=root/'ground_truth'/kind/(path.stem+'_mask.png')
    heatmap=run/'official/mvtec_ad/ViT-L-14-336/imagesize518/bottle'/kind/path.name
    for col,p in enumerate([path,mask,heatmap]):
        canvas.paste(Image.open(p).convert('RGB').resize((300,300)),(col*300,30+row*310))
    draw.text((5,330+row*310),kind+'/'+path.name,fill='black')
    manifest.append(dict(image=str(path),mask=str(mask),heatmap=str(heatmap)))
canvas.save(out/'preview.png')
(out/'preview_manifest.json').write_text(json.dumps(manifest,indent=2))
