"""Explicit fixed-last training entry. This script is not run by package preparation."""
import argparse,json,sys,subprocess,hashlib,contextlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CATEGORIES={'mvtec_ad2':['can','sheet_metal','vial','wallplugs'],'visa':['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum'],'btad':['01','02','03']}
class Tee:
 def __init__(self,a,b):self.a,self.b=a,b
 def write(self,s):self.a.write(s);self.b.write(s);self.b.flush();return len(s)
 def flush(self):self.a.flush();self.b.flush()
def main():
 p=argparse.ArgumentParser(description='Train a single category or all preregistered categories sequentially; never select a test-best checkpoint.')
 p.add_argument('--dataset',choices=list(CATEGORIES),required=True);p.add_argument('--category',required=True,help='Category name or all')
 p.add_argument('--data-root',required=True);p.add_argument('--output-root',required=True);p.add_argument('--variant',default='gs_hf_full',choices=['baseline','gs_only','hf_only','gs_hf_full'])
 p.add_argument('--seed',type=int,default=42)
 args=p.parse_args()
 if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
 from utils.config import load_training_config
 from efficientad import run_training
 cats=CATEGORIES[args.dataset] if args.category=='all' else [args.category]
 if any(c not in CATEGORIES[args.dataset] for c in cats):p.error('Category not in the frozen protocol')
 data=Path(args.data_root).resolve();out=Path(args.output_root).resolve()
 if not data.is_dir():raise FileNotFoundError('Provide the official dataset under --data-root')
 teacher=ROOT/'models/teacher_small.pth'
 if not teacher.is_file():raise FileNotFoundError('Pretrained teacher is required; see models/README.md. Random-teacher fallback is forbidden.')
 targets=[out/args.variant/c/('seed'+str(args.seed)) for c in cats]
 if any(x.exists() for x in targets):raise FileExistsError('A target run exists. Keep it and choose a new --output-root; no overwrite or automatic retry.')
 for cat,target in zip(cats,targets):
  c=load_training_config(str(ROOT/f'configs/{args.dataset}/ghead.yaml'),ROOT,overrides={'dataset':{'category':cat,'dataset_root':str(data)},'basic':{'output_root':str(out),'save_dir':str(target),'seed':args.seed},'model':{'variant':args.variant}})
  assert c.seed==args.seed and c.train_steps==70000 and c.save_top_k==0 and not c.run_test_after_fit
  target.mkdir(parents=True,exist_ok=False)
  (target/'train_command.txt').write_text(subprocess.list2cmdline([sys.executable,*sys.argv])+'\n',encoding='utf-8')
  (target/'public_launch_config.json').write_text(json.dumps(c.raw_config,indent=2),encoding='utf-8')
  with (target/'launcher.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
   try:
    summary=run_training(c)
    ck=target/'checkpoints/last';meta=json.loads((ck/'metadata.json').read_text())
    assert meta['step']==70000 and meta['type']=='last'
    for name in ['teacher.pth','student.pth','autoencoder.pth']:
     if not (ck/name).is_file():raise FileNotFoundError(ck/name)
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ck.iterdir() if f.is_file()}
    (target/'final_bundle_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
   except Exception:
    traceback.print_exc();raise
if __name__=='__main__':main()
