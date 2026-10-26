"""Explicit run directories, complete test split, fixed last checkpoint only."""
import argparse,json,sys,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CATEGORIES={'mvtec_ad2':['can','sheet_metal','vial','wallplugs'],'visa':['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum'],'btad':['01','02','03']}
def main():
 p=argparse.ArgumentParser(description='Evaluate a fixed 70000-step last bundle; no latest/best fallback, no test subsets.')
 p.add_argument('--dataset',choices=list(CATEGORIES),required=True);p.add_argument('--category',required=True,help='Category or all')
 p.add_argument('--data-root',required=True);p.add_argument('--output-root',required=True);p.add_argument('--run-dir');p.add_argument('--run-root')
 p.add_argument('--variant',default='gs_hf_full',choices=['baseline','gs_only','hf_only','gs_hf_full']);p.add_argument('--device',default='cuda:0')
 a=p.parse_args();cats=CATEGORIES[a.dataset] if a.category=='all' else [a.category]
 if any(c not in CATEGORIES[a.dataset] for c in cats):p.error('Category not in the frozen protocol')
 if bool(a.run_dir)==bool(a.run_root):p.error('Supply exactly one of --run-dir or --run-root')
 if a.run_dir and len(cats)!=1:p.error('--run-dir is for a single category')
 data=Path(a.data_root).resolve();output=Path(a.output_root).resolve()
 if not data.is_dir():raise FileNotFoundError(data)
 plans=[]
 for cat in cats:
  run=Path(a.run_dir).resolve() if a.run_dir else Path(a.run_root).resolve()/a.variant/cat/'seed42'
  ck=run/'checkpoints/last';m=json.loads((ck/'metadata.json').read_text())
  if m.get('type')!='last' or m.get('step')!=70000:raise ValueError('Invalid fixed-last bundle')
  if m.get('model_variant')!=a.variant:raise ValueError('Declared variant does not match checkpoint metadata')
  for n in ['teacher.pth','student.pth','autoencoder.pth']:
   if not (ck/n).is_file():raise FileNotFoundError(ck/n)
  dest=output/cat
  if dest.exists():raise FileExistsError('Evaluation destination exists; select a new --output-root')
  plans.append((cat,run,ck,dest))
 for cat,run,ck,dest in plans:
  dest.mkdir(parents=True,exist_ok=False)
  argv=[sys.executable,'-B',str(ROOT/'tools/evaluate_model.py'),'--config',str(ROOT/f'configs/{a.dataset}/ghead.yaml'),'--variant',a.variant,'--category',cat,'--dataset_root',str(data),'--run-dir',str(run),'--checkpoint-kind','last','--split','test_public' if a.dataset=='mvtec_ad2' else 'test','--device',a.device,'--save-dir',str(dest)]
  (dest/'evaluation_command.txt').write_text(subprocess.list2cmdline(argv)+'\n',encoding='utf-8')
  before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ck.iterdir() if f.is_file()}
  (dest/'source_checkpoint_hashes.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
  with (dest/'evaluation.log').open('x',encoding='utf-8') as log:
   result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
  if result.returncode:raise RuntimeError(f'Evaluation failed: {cat}; inspect the preserved log')
  after={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ck.iterdir() if f.is_file()}
  if before!=after:raise RuntimeError('Checkpoint hash changed')
  print('FINAL EVALUATION COMPLETE:',a.dataset,cat)
if __name__=='__main__':main()
