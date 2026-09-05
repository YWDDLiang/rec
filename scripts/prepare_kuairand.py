import argparse,json
from rec_lab.data import prepare_kuairand
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--logs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-end-ms',type=float,required=True);p.add_argument('--selection-end-ms',type=float,required=True)
    p.add_argument('--max-rows',type=int,default=200000);p.add_argument('--history',type=int,default=20);a=p.parse_args()
    print(json.dumps(prepare_kuairand(a.logs,a.output,a.train_end_ms,a.selection_end_ms,a.history,a.max_rows),indent=2))
