"""Convert frozen SFT prompt/answer JSONL to the shared causal encoding contract.

This generic HF-chat path is not an upstream-template reproduction. Compare
encoded IDs/masks against the upstream trainer before using its paper protocol.
"""
import argparse,json
from pathlib import Path
from rec_pool.contracts import Record,write_records,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--template-kwargs',default='{}');p.add_argument('--max-length',type=int,default=2048)
    args=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    if not tok.chat_template:raise ValueError('no chat template: export native production IDs instead')
    kwargs=json.loads(args.template_kwargs)
    recipe=digest({'tokenizer_path':str(Path(args.tokenizer).resolve()),'vocab':tok.get_vocab(),
                   'template':tok.chat_template,'template_kwargs':kwargs,'max_length':args.max_length})
    rows=[]
    for line in Path(args.input).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        d=json.loads(line);messages=[]
        if d.get('system'):messages.append({'role':'system','content':d['system']})
        messages.append({'role':'user','content':d['input_text']})
        prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,**kwargs)
        full=tok.apply_chat_template(messages+[{'role':'assistant','content':d['output_text']}],tokenize=True,add_generation_prompt=False,**kwargs)
        if list(full[:len(prefix)])!=list(prefix):raise ValueError('template prefix mismatch; use native production encoder')
        if len(full)>args.max_length:raise ValueError('overlength sample; truncate explicitly in upstream recipe, do not silently drop')
        labels=[-100]*len(prefix)+list(full[len(prefix):]);labels[0]=-100
        rows.append(Record(record_id=d['record_id'],root_id=d['root_id'],task=d['task'],domain=d['domain'],
             source=d['source'],split=d['split'],input_ids=tuple(full),labels=tuple(labels),recipe_id=recipe,
             user_id=d.get('user_id'),label_origin=d.get('label_origin','observed')))
    write_records(rows,args.output)


if __name__=='__main__':main()
