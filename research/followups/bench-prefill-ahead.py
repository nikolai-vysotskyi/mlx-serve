"""Matched HTTP screening on one frozen executable. Does not replace llmprobe."""
import os,json,time,subprocess,hashlib,urllib.request,re,socket,argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,default=Path('zig-out/bin/mlx-serve'))
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--lock',type=Path,required=True,help='exclusive GPU lock implementing acquire/release OWNER')
args=parser.parse_args()
P=Path(__file__).resolve().parents[2]
W=args.out.resolve();W.mkdir(parents=True,exist_ok=True)
BIN=args.binary.resolve()
MODEL='ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit'
PORT=11234
def listening():
 with socket.socket() as sock:
  sock.settimeout(.2)
  return sock.connect_ex(('127.0.0.1',PORT))==0
TAG=os.environ.get('PLE_BENCH_TAG','ple-packed-ab')
corpus='\n'.join((P/'src/transformer.zig').read_text().splitlines()[100:900])
records=[]
meta={'kind':'HTTP matched screening; not llmprobe','binary_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),'power':subprocess.check_output(['pmset','-g','batt'],text=True),'corpus_sha256':hashlib.sha256(corpus.encode()).hexdigest(),'records':records}
out=W/(TAG+'.json')
def save():out.write_text(json.dumps(meta,indent=2)+'\n')
def props():
 with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/props') as r:return json.load(r).get('memory',{})
def request(prompt):
 body={'model':MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':1234,'max_tokens':32,'enable_mtp':False,'enable_pld':False,'stream':False}
 t=time.monotonic()
 with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as r:response=json.load(r)
 return {'wall_s':time.monotonic()-t,'response':response,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
if listening():raise RuntimeError('port occupied')
owner='ple-packed-ab-'+str(os.getpid())
subprocess.run([str(args.lock.resolve()),'acquire',owner],check=True)
try:
 for name,fused,packed in [('base',False,False),('current',True,False),('packed',True,True),('ahead',True,True)]:
  if name not in os.environ.get('PLE_BENCH_ARMS','base,current,packed').split(','):continue
  env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(P/'lib/mlx/lib')+':'+str(P/'lib/llama/lib'),MLX_SERVE_PREFILL_CHUNK='8192',MLX_SERVE_PREFILL_TRACE='1',MLX_SERVE_QSA_PAIR=str(int(fused)),MLX_SERVE_HC_PREFILL=str(int(fused)),MLX_SERVE_GDN_PREFILL_FUSED=str(int(fused)),MLX_SERVE_PLE_PACKED=str(int(packed)),MLX_SERVE_PLE_AHEAD=str(int(name=='ahead')),MLX_SERVE_PLE_TIMING='1',MLX_SERVE_MOE_PREFILL_GROUP='0')
  env.pop('QWEN4_PROFILE_FWD',None)
  if os.environ.get('PLE_BENCH_CPU_TIMING')=='1':env['QWEN4_PROFILE_FWD']='1'
  logpath=W/(TAG+'-'+name+'.log')
  with logpath.open('w') as log:
   start=time.monotonic();server=subprocess.Popen([str(BIN),'serve','--host','127.0.0.1','--port',str(PORT),'--ctx-size','131072','--prefix-cache-entries','0','--prefill-chunk','8192','--kv-quant','off'],env=env,stdout=log,stderr=log)
   try:
    while not listening():
     if server.poll() is not None:raise RuntimeError(logpath.read_text()[-4000:])
     if time.monotonic()-start>30:raise TimeoutError('bind')
     time.sleep(.2)
    ready=request('Capital of France? Answer with one word.')
    assert 'paris' in ready['response']['choices'][0]['message']['content'].lower(),ready
    arm={'arm':name,'loaded_s':time.monotonic()-start,'memory_before':props(),'requests':[]}
    records.append(arm);save();print(json.dumps({k:v for k,v in arm.items() if k!='requests'}),flush=True)
    for nonce,size in ([('81492017',40000),('52839106',185000)] if os.environ.get('PLE_BENCH_LONG')=='1' else [('81492017',40000),('52839106',47000)]):
     prompt=f'Nonce {nonce}. Remember the passphrase MAGNOLIA-7731.\n'+(corpus*(16 if os.environ.get('PLE_BENCH_LONG')=='1' else 4))[:size]+'\nWhat was the passphrase? Answer with the passphrase only.'
     offset=logpath.stat().st_size;r=request(prompt);time.sleep(.1)
     emitted=logpath.read_bytes()[offset:].decode(errors='replace')
     rates=re.findall(r'prefill: ([0-9.]+) tok/s, decode: ([0-9.]+) tok/s',emitted)
     r.update(prefill_tps=float(rates[-1][0]) if rates else None,decode_tps=float(rates[-1][1]) if rates else None,memory_after=props(),server_log=emitted)
     arm['requests'].append(r);save()
     response=r['response'];usage=response['usage'];answer=response['choices'][0]['message']['content']
     print(json.dumps({'arm':name,'prefill_tps':r['prefill_tps'],'decode_tps':r['decode_tps'],'usage':usage,'answer':answer,'memory_after':r['memory_after']}),flush=True)
     assert usage.get('prompt_tokens_details',{}).get('cached_tokens',0)==0,usage
     assert 'MAGNOLIA-7731' in answer,response
    text=logpath.read_text()
    assert ('[ple-packed] engaged:' in text)==packed
    if name=='ahead':assert '[ple-ahead] chunks=' in text and 'chunks=0/' not in text,text[-4000:]
    if fused:assert '[qsa-pair] engaged:' in text and '[hc-prefill] engaged:' in text and '[gdn-prefill] engaged:' in text
   finally:
    server.terminate()
    try:server.wait(timeout=20)
    except subprocess.TimeoutExpired:server.kill();server.wait()
    deadline=time.monotonic()+10
    while listening() and time.monotonic()<deadline:time.sleep(.2)
    if listening():raise RuntimeError('owned port did not close')
finally:subprocess.run([str(args.lock.resolve()),'release',owner],check=True)
