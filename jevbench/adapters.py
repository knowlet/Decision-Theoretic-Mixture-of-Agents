"""Pinned native decision heads with CPU-only checkpoint loaders.
No API inference, generated labels, random-head fallback, or model training.
Native input layouts differ but receive identical semantic requests.
"""
from __future__ import annotations
import importlib.util,json,hashlib,sys,time
from pathlib import Path
import numpy as np
MODELS={
 'nanojev':('C-Tianyu/NanoJev','4a19595eada0857133c0d2be024f879a4077054b'),
 'decider':('Mapika/decider-2b','1d96be0093133e194fe18105a521b3e69be931d2'),
 'system_one':('pngwn/system-one-qwen3.5-4b-scorer','e6464dce15f013c2ef641593a85cc6afcdaea928')}
BASE=('Qwen/Qwen3.5-4B-Base','1001bb4d826a52d1f399e183466143f4da7b741b')
NANO_SOURCE=('TianyuCodings/NanoJev','71a513bb0163b5634467842b523ee0c0ed6fb1c7')
MAX_TOKENS=2048

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError('Module unavailable')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def cpu_accumulation():
    """Disclosed BF16 storage / FP32 Linear-Conv accumulation compatibility backend."""
    import torch,torch.nn.functional as F
    if getattr(torch.nn.Linear,'_dtmoa_patched',False):return
    linear=torch.nn.Linear.forward;conv=torch.nn.Conv1d.forward
    def forward(self,x):
        if x.device.type=='cpu' and self.weight.dtype==torch.bfloat16:
            return F.linear(x.float(),self.weight.float(),None if self.bias is None else self.bias.float()).to(x.dtype)
        return linear(self,x)
    def conv_forward(self,x):
        if x.device.type=='cpu' and self.weight.dtype==torch.bfloat16:
            return self._conv_forward(x.float(),self.weight.float(),None if self.bias is None else self.bias.float()).to(x.dtype)
        return conv(self,x)
    torch.nn.Linear.forward=forward;torch.nn.Conv1d.forward=conv_forward;torch.nn.Linear._dtmoa_patched=True

def snapshot(key):
    from huggingface_hub import snapshot_download
    repo,rev=MODELS[key]
    patterns=(['best.safetensors','config.json','tokenizer/*','backbone_config/*'] if key=='nanojev'
              else ['*.json','*.safetensors','*.py','*.txt','*.model','decider/*.py'])
    return Path(snapshot_download(repo,revision=rev,allow_patterns=patterns,max_workers=2))

class Adapter:
    def __init__(self,name,vendor=Path('vendor/nanojev')):
        import torch,transformers
        from safetensors.torch import load_file
        torch.manual_seed(160918);torch.set_num_threads(4)
        self.name=name;self.torch=torch;self.path=snapshot(name);self.vendor=Path(vendor)
        started=time.perf_counter();self.temperature=1.;self.code={}
        if name=='nanojev':
            api=load_module(self.vendor/'scripts/predict_toy_decisions.py','_dtmoa_nano_api')
            trainer=load_module(self.vendor/'scripts/train_toy_decisions.py','_dtmoa_nano_train')
            cfg=json.loads((self.path/'config.json').read_text())
            self.tok=transformers.AutoTokenizer.from_pretrained(str(self.path/'tokenizer'),local_files_only=True,trust_remote_code=False)
            bc=transformers.AutoConfig.from_pretrained(str(self.path/'backbone_config'),local_files_only=True,trust_remote_code=False);bc.use_cache=False
            body=transformers.AutoModel.from_config(bc,attn_implementation='sdpa',trust_remote_code=False).float()
            self.model=trainer.DecisionModel(body,cfg['set_head'])
            state=load_file(str(self.path/'best.safetensors'),device='cpu');self.model.load_state_dict(state,strict=True);del state
            self.model.float().eval();self.api=api
            self.code={p:sha(self.vendor/p) for p in ['scripts/predict_toy_decisions.py','scripts/train_toy_decisions.py']}
            self.dtype='float32';self.layout='upstream choice paths and attention set head; root navigation checkpoint'
        elif name=='decider':
            sys.path.insert(0,str(self.path))
            from decider.infer import Decider
            cpu_accumulation();self.api=Decider(str(self.path),device='cpu',dtype=torch.bfloat16,use_graphs=False)
            self.model=self.api.m;self.tok=self.model.tok;self.model.eval();self.model.lm.config.use_cache=False
            self.temperature=self.api.T;self.code={str(p.relative_to(self.path)):sha(p) for p in (self.path/'decider').glob('*.py')}
            self.dtype='bfloat16';self.layout='upstream state-first decide, no CUDA graphs, original option normalization'
        elif name=='system_one':
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            cpu_accumulation()
            base=Path(snapshot_download(BASE[0],revision=BASE[1],allow_patterns=['*.json','*.safetensors','*.model','*.txt'],max_workers=2))
            self.api=load_module(self.path/'system_one.py','_dtmoa_system_one')
            self.tok=transformers.AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
            if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
            model,info=transformers.Qwen3_5TextForSequenceClassification.from_pretrained(str(base),num_labels=1,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,local_files_only=True,trust_remote_code=False,output_loading_info=True)
            bad=[k for k in info.get('missing_keys',[]) if k!='score.weight']
            if bad or info.get('mismatched_keys') or info.get('error_msgs'):raise RuntimeError('Incomplete base checkpoint: '+str(info))
            model.config.pad_token_id=self.tok.pad_token_id;model.config.eos_token_id=self.tok.eos_token_id
            tc=model.config.get_text_config();tc.pad_token_id=self.tok.pad_token_id;tc.eos_token_id=self.tok.eos_token_id;tc.num_labels=1;tc.use_cache=False
            model.num_labels=1;model.config.use_cache=False
            self.model=PeftModel.from_pretrained(model,str(self.path),is_trainable=False).to(dtype=torch.bfloat16).eval()
            weights=load_file(str(self.path/'adapter_model.safetensors'),device='cpu')
            headkeys=[k for k in weights if k.endswith('score.weight') or k.endswith('score.modules_to_save.weight')]
            live=[(n,p) for n,p in self.model.named_parameters() if 'score.modules_to_save.default.weight' in n]
            if len(headkeys)!=1 or len(live)!=1:raise RuntimeError('Cannot verify saved scoring head: '+str((headkeys,[n for n,_ in live])))
            if not torch.equal(live[0][1].detach().cpu(),weights[headkeys[0]].to(torch.bfloat16)):raise RuntimeError('Saved classifier head mismatch')
            del weights;self.temperature=1.75
            self.code={'system_one.py':sha(self.path/'system_one.py'),'adapter_config.json':sha(self.path/'adapter_config.json')}
            self.dtype='bfloat16';self.layout='upstream option head and renderer, options batched, publisher T=1.75'
        else:raise ValueError(name)
        self.model.eval()
        self.meta={'id':MODELS[name][0],'revision':MODELS[name][1],'dtype':self.dtype,'device':'cpu','torch_threads':4,
            'temperature':self.temperature,'layout':self.layout,'torch':torch.__version__,'transformers':transformers.__version__,
            'load_seconds_excluding_download':time.perf_counter()-started,'code_sha256':self.code,
            'backend':'Native FP32 NanoJev; BF16 storage/FP32 Linear-Conv accumulation for Qwen3.5; no GPU parity claim',
            'adapter_sha256':sha(__file__),'new_training_steps':0,'parameters':sum(p.numel() for p in self.model.parameters())}
    def score(self,request):
        """Semantic fixture only, no evaluation label."""
        opts=request['options'];context=request['state'];question=request['question']
        if not 2<=len(opts)<=255 or len(set(o['id'] for o in opts))!=len(opts):raise ValueError('Invalid candidates')
        texts=[o['text'] for o in opts];started=time.perf_counter();torch=self.torch
        if self.name=='nanojev':
            payload={'states':[{'id':request['id'],'state':context,'questions':{'answer':{'type':'choice','instructions':question,'criteria':{o['id']:o['text'] for o in opts}}}}]}
            examples=self.api.prepare_examples(payload,self.tok,MAX_TOKENS)
            lengths=[len(x) for x in examples[0]['leaf_tokens']];prep=time.perf_counter()-started
            with torch.inference_mode():logits,_=self.model(examples,self.tok.pad_token_id)
            values=logits[0,:len(opts)].float().cpu().tolist();paths=len(opts)
        elif self.name=='decider':
            from decider.infer import Example,Q,neutralize_options
            from decider.prompt import build,MAX_OPTIONS
            class Keep:
                def shuffle(self,x):pass
                def sample(self,x,k):return x[:k]
            normalized=neutralize_options(texts)[0] if self.api.neutralize_none else texts
            item=build(Example(context,[Q(question,normalized,0)],'infer'),self.tok,Keep(),max_options=MAX_OPTIONS,max_ctx_tokens=1000000)
            if len(item['ids'])>MAX_TOKENS:raise ValueError('Input exceeds common limit; not truncated')
            lengths=[len(item['ids'])];prep=time.perf_counter()-started
            from decider.model import collate
            batch=collate([item],self.tok.pad_token_id)
            with torch.inference_mode():logits=self.model.slot_logits(batch['input_ids'],batch['attention_mask'],batch['slot_idx'],batch['slot_batch'],batch['nopts'])
            values=logits[0,:len(opts)].float().cpu().tolist();paths=1
        else:
            seqs=[self.api.encode(self.tok,context,question,x,1000000) for x in texts];lengths=list(map(len,seqs))
            if max(lengths)>MAX_TOKENS:raise ValueError('Input exceeds common limit; not truncated')
            ids,mask=self.api.pad_batch(seqs,self.tok.pad_token_id,MAX_TOKENS);prep=time.perf_counter()-started
            with torch.inference_mode():values=self.model(input_ids=ids,attention_mask=mask).logits.squeeze(-1).float().cpu().tolist()
            paths=len(opts)
        x=np.array(values,dtype=float)/self.temperature
        if not np.isfinite(x).all():raise RuntimeError('Nonfinite native logits')
        probs=np.exp(x-x.max());probs/=probs.sum()
        return {'id':request['id'],'option_ids':[o['id'] for o in opts],'probabilities':probs.tolist(),'raw_logits':values,
                'total_ms':1000*(time.perf_counter()-started),'preprocess_ms':1000*prep,'input_tokens_sum':sum(lengths),
                'max_path_tokens':max(lengths),'candidate_paths':paths,'backbone_calls':1,'decode_steps':0}
