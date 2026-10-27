"""Offline Track 1 runtime: the members listed in assets/config.json, equal-weight logit fusion, PMCD decoding."""
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
CONFIG=json.loads((ROOT/'assets/config.json').read_text())
T0=None
BUDGET_SECONDS=100000
INTERNAL_BATCH_SIZE=32
DECODE_WORKERS=12
STATE=None
GROUPS=((0,1,2),(4,5,6),tuple(range(8,20)),tuple(range(21,33)),(33,34))


def plogit(p):
    p=np.asarray(p,np.float64)
    if not np.isfinite(p).all() or ((p<=0)|(p>=1)).any():
        raise ValueError('TRAIN priors must be finite and strictly between zero and one')
    return np.log(p)-np.log1p(-p)


def fuse_raw(raw):
    if len(raw)!=len(CONFIG['members']):
        raise ValueError('Exactly four ordered S2 logits required')
    arrays=[np.asarray(z,np.float64) for z in raw]
    if any(z.ndim!=2 or z.shape!=arrays[0].shape or z.shape[1]!=40 or not np.isfinite(z).all() for z in arrays):
        raise ValueError('Aligned finite [N,40] raw logits required')
    alpha=.75 if CONFIG['track1']['decoder']=='alpha075' else 1.
    return np.mean([z-alpha*plogit(m['train_prior']) for z,m in zip(arrays,CONFIG['members'])],axis=0)


def decode_fused(z):
    z=np.asarray(z,np.float64)
    if CONFIG['track1']['decoder']=='pmcd':
        from pmcd import pmcd_probabilities
        p=np.asarray(CONFIG['track1']['pmcd_reference_prior'],np.float64)
        return pmcd_probabilities(z+plogit(p),p)
    decisions=z>=0
    for ix in GROUPS:
        ix=np.asarray(ix)
        rows=np.flatnonzero(~decisions[:,ix].any(1))
        decisions[rows,ix[z[rows][:,ix].argmax(1)]]=True
    probs=(1/(1+np.exp(-np.clip(z,-80,80)))).astype(np.float32)
    probs[decisions]=np.maximum(probs[decisions],np.nextafter(np.float32(.5),np.float32(1)))
    probs[~decisions]=np.minimum(probs[~decisions],np.nextafter(np.float32(.5),np.float32(0)))
    return probs


def probabilities_from_raw(raw):
    return decode_fused(fuse_raw(raw))


class SiglipTower(nn.Module):
    def __init__(self,vision):
        super().__init__(); self.vision=vision
    def forward(self,x):
        return self.vision(x)


def load(flip_tta=None,rate_preserving=None):
    global STATE,T0
    # Start once at the public load_model boundary, including model loading.
    if T0 is None:
        T0=time.monotonic()
    if flip_tta or rate_preserving:
        raise ValueError('UF4 locks flip/RP off')
    if STATE is not None:
        return STATE
    torch.set_num_threads(int(os.environ.get('UPAR_THREADS','2')))
    device='cuda' if torch.cuda.is_available() and os.environ.get('UPAR_FORCE_CPU')!='1' else 'cpu'
    precision=os.environ.get('UPAR_UF4_WEIGHT_DTYPE',CONFIG.get('runtime',{}).get('default_weight_dtype','float32'))
    if precision not in ('float32','float16'):
        raise ValueError('Only FP32 or parity-verified FP16 runtime weights are supported')
    dtype=torch.float16 if precision=='float16' and device=='cuda' else torch.float32
    models=[]
    for member in CONFIG['members']:
        key=member['name']; spec=member['spec']
        if key.startswith('sl2'):
            from siglip_pure import SiglipVision
            vision=SiglipVision(**member['vision_config'])
            tower=SiglipTower(vision)
        else:
            from teacher_system.pe import VisionTransformer
            width,layers,heads=(768,12,12) if key=='peb' else (1024,24,16)
            tower=VisionTransformer(patch_size=spec['patch'],width=width,layers=layers,heads=heads,
                mlp_ratio=4,output_dim=1024,image_size=spec['native'],use_cls_token=True)
        state=torch.load(ROOT/member['checkpoint'],map_location='cpu',weights_only=True)
        tower.load_state_dict(state['tower_state_dict'],strict=True)
        head=nn.Linear(state['head_state_dict']['weight'].shape[1],40)
        head.load_state_dict(state['head_state_dict'],strict=True)
        models.append((tower.to(device=device,dtype=dtype).eval(),head.to(device=device,dtype=dtype).eval()))
        del state
    fallback=probabilities_from_raw([plogit(m['train_prior'])[None,:] for m in CONFIG['members']])[0]
    # LUT exactly reproduces the original CPU float32 normalization for every
    # possible uint8 intensity. Lookup happens on GPU, avoiding div/FMA drift.
    lut=((torch.arange(256,dtype=torch.float32)/255.)-.5)/.5
    STATE=dict(models=models,device=device,dtype=dtype,prior=fallback,fallback_rows=0,
               done_total=0,normalization_lut=lut.to(device),
               pool=ThreadPoolExecutor(max_workers=DECODE_WORKERS),guard_exhausted=False)
    return STATE


def _member_hw(member):
    if 'input_hw' in member:
        return tuple(int(x) for x in member['input_hw'])
    p=int(member['spec']['patch'])
    return (16*p,8*p)


def _preprocess(path,hw):
    h,w=(hw if isinstance(hw,(tuple,list)) else (16*int(hw),8*int(hw)))
    with Image.open(path) as image:
        image=image.convert('RGB').resize((w,h),Image.Resampling.BILINEAR)
        x=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255.
    return (x-torch.tensor([.5,.5,.5])[:,None,None])/torch.tensor([.5,.5,.5])[:,None,None]


def _decode_uint8(path):
    """One JPEG decode, same RGB/bilinear squash for each member input_hw."""
    hws={_member_hw(m) for m in CONFIG['members']}
    with Image.open(path) as source:
        image=source.convert('RGB')
        return {hw:np.asarray(image.resize((hw[1],hw[0]),Image.Resampling.BILINEAR),
                              dtype=np.uint8).copy().transpose(2,0,1)
                for hw in hws}


def normalize_uint8(pixels,lut,device):
    x=torch.from_numpy(np.ascontiguousarray(pixels))
    if device=='cuda':
        x=x.pin_memory().to(device,non_blocking=True)
    else:
        x=x.to(device)
    return lut[x.long()]


def _remaining():
    if CONFIG.get('track1',{}).get('disable_guard',False):
        return 100000.0
    return BUDGET_SECONDS-(time.monotonic()-T0)


def _save_guard_result(state,output,paths,start):
    """Persist the current public call immediately; ingestion owns its CSV."""
    output[start:]=state['prior']
    state['fallback_rows']+=len(paths)-start
    state['guard_exhausted']=True
    state['pool'].shutdown(wait=False,cancel_futures=True)
    report=dict(status='TIME_GUARD_EXHAUSTED',guard_seconds=BUDGET_SECONDS,
        elapsed_seconds=time.monotonic()-T0,model_rows_total=state['done_total'],
        fallback_rows_total=state['fallback_rows'],call_rows=len(paths),
        real_model_rows_in_call=start,complete_call_saved=False,
        note='Emergency fallback is never a valid parity/runtime PASS')
    location=ROOT/'guard_result.json'
    temp=ROOT/'guard_result.json.tmp'
    temp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');temp.replace(location)
    if not state.get('guard_snapshot_saved'):
        data=ROOT/'guard_predictions.npz'
        partial=ROOT/'guard_predictions.npz.tmp'
        with partial.open('wb') as f:
            np.savez(f,probabilities=output,image_paths=np.asarray(paths),model_rows=np.asarray(start))
        partial.replace(data)
        state['guard_snapshot_saved']=True
        report.update(complete_call_saved=True,predictions_file=str(data))
        temp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');temp.replace(location)


def column_index(names):
    own=CONFIG['attribute_names']
    if len(names)!=40 or set(names)!=set(own):
        raise ValueError('Exactly the organizer attributes required')
    return [own.index(name) for name in names]


@torch.inference_mode()
def probabilities(paths,tag='track1'):
    state=load(); output=np.empty((len(paths),40),np.float32)
    if not paths:return output
    if state['guard_exhausted'] or _remaining()<=0:
        _save_guard_result(state,output,paths,0)
        return output
    def submit(start):
        return [state['pool'].submit(_decode_uint8,p) for p in paths[start:start+INTERNAL_BATCH_SIZE]]
    pending=submit(0)
    for start in range(0,len(paths),INTERNAL_BATCH_SIZE):
        if _remaining()<=0:
            for f in pending:f.cancel()
            _save_guard_result(state,output,paths,start);break
        try:
            decoded=[f.result(timeout=max(.001,_remaining())) for f in pending]
        except FutureTimeout:
            for f in pending:f.cancel()
            _save_guard_result(state,output,paths,start);break
        next_start=start+INTERNAL_BATCH_SIZE
        pending=submit(next_start) if next_start<len(paths) and _remaining()>0 else []
        if _remaining()<=0:
            for f in pending:f.cancel()
            _save_guard_result(state,output,paths,start);break
        hws={_member_hw(m) for m in CONFIG['members']}
        preps={hw:normalize_uint8(np.stack([item[hw] for item in decoded]),
                    state['normalization_lut'],state['device']) for hw in hws}
        raw=[]
        for member,(tower,head) in zip(CONFIG['members'],state['models']):
            amp=torch.autocast('cuda',dtype=torch.float16) if state['device']=='cuda' else contextlib.nullcontext()
            with amp:
                z=head(tower(preps[_member_hw(member)]))
            raw.append(z.float().cpu().numpy())
        output[start:start+len(decoded)]=probabilities_from_raw(raw)
        state['done_total']+=len(decoded)
        # Check after forwards too, including on the final batch.
        if _remaining()<=0:
            for f in pending:f.cancel()
            _save_guard_result(state,output,paths,start+len(decoded));break
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite ensemble output')
    return output
