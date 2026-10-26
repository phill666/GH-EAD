"""Thin isolated factory switch; default main model remains unchanged."""
import argparse,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description='GSH structural sensitivity: retraining required; not inference-only.')
 p.add_argument('--groups',type=int,choices=[1,2,3,4],default=2);p.add_argument('--half-ratio',type=float,choices=[.25,.5,.75],default=.5)
 p.add_argument('--no-shuffle',action='store_true');p.add_argument('--no-se',action='store_true');a,rest=p.parse_known_args()
 sys.path.insert(0,str(ROOT))
 from modules.gsh_switches import install_gsh_switches
 install_gsh_switches({'groups':a.groups,'half_ratio':a.half_ratio,'shuffle':not a.no_shuffle,'se_gate':not a.no_se})
 sys.argv=[str(ROOT/'scripts/train/train.py'),*rest]
 runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
