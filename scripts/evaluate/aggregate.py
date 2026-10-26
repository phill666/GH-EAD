"""CPU-only category-equal aggregation; rejects incomplete or mixed-variant inputs."""
import argparse,csv,math,json,hashlib
from pathlib import Path
CATEGORIES={'mvtec_ad2':['can','sheet_metal','vial','wallplugs'],'visa':['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum'],'btad':['01','02','03']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CATEGORIES),required=True);p.add_argument('--input-root',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
 rows=[];sources=[];variants=set();splits=set()
 for cat in CATEGORIES[a.dataset]:
  path=Path(a.input_root)/cat/'metrics.csv'
  with path.open(encoding='utf-8-sig',newline='') as f:data=list(csv.DictReader(f))
  if len(data)!=1 or data[0]['category']!=cat or data[0]['checkpoint_kind']!='last':raise ValueError('Invalid category/final checkpoint metadata')
  r=data[0];variants.add(r['variant']);splits.add(r['split'])
  v=[float(r[k]) for k in ['image_auroc','pixel_auroc','au_pro']]
  if not all(math.isfinite(x) and 0<=x<=100 for x in v):raise ValueError('Invalid metrics')
  rows.append({'Category':cat,'Method':r['variant'],'Image AUROC':v[0],'Pixel AUROC':v[1],'AU-PRO':v[2]})
  sources.append({'category':cat,'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 if len(variants)!=1 or splits!={'test_public' if a.dataset=='mvtec_ad2' else 'test'}:raise ValueError('Mixed methods or splits cannot be aggregated')
 macro={'Category':'Macro Average','Method':next(iter(variants)),**{k:sum(r[k] for r in rows)/len(rows) for k in ['Image AUROC','Pixel AUROC','AU-PRO']}}
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 for name,values in [('per_category.csv',rows),('macro.csv',[macro])]:
  with (out/name).open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(macro));w.writeheader();w.writerows(values)
 (out/'metric_sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
 print(json.dumps(macro))
if __name__=='__main__':main()
