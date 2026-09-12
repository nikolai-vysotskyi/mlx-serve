"""Matched HTTP screening on one frozen executable. Does not replace llmprobe."""
import os,json,time,subprocess,hashlib,urllib.request,re,socket,argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,default=Path('zig-out/bin/mlx-serve'))
parser.add_argument('--trace',action='store_true',help='diagnostic Metal System Trace; rates then include tracing overhead')
parser.add_argument('--sequence',help='same-process arms, one digit per identical request: 0=current, 1=group, 2=PLE ahead, 3=combined')
parser.add_argument('--monitor',type=Path,help='optional local macmon executable; read-only 1-second telemetry')
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--lock',type=Path,required=True,help='exclusive GPU lock implementing acquire/release OWNER')
args=parser.parse_args()
if args.sequence and (not set(args.sequence)<=set('0123') or args.trace):parser.error('sequence accepts only 0..3 and cannot use trace')
P=Path(__file__).resolve().parents[2]
W=args.out.resolve();W.mkdir(parents=True,exist_ok=True)
BIN=args.binary.resolve()
MODEL='ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit'
PORT=11234
def listening():
 with socket.socket() as sock:
  sock.settimeout(.2)
  return sock.connect_ex(('127.0.0.1',PORT))==0
TAG=os.environ.get('PLE_BENCH_TAG','combined-cadence')
corpus='\n'.join((P/'src/transformer.zig').read_text().splitlines()[100:900])
records=[]
diagnostic=args.trace or os.environ.get('PREFILL_CAPTURE_ROUTES')=='1' or os.environ.get('PREFILL_MOE_LIVE_AB')=='1'
if args.sequence and diagnostic:parser.error('sequence must not add route capture or layer replay synchronization')
meta={'kind':('diagnostic; synchronization/trace overhead included' if diagnostic else 'HTTP matched screening; not llmprobe'),'binary_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),'power':subprocess.check_output(['pmset','-g','batt'],text=True),'corpus_sha256':hashlib.sha256(corpus.encode()).hexdigest(),'records':records}
meta['sequence']=args.sequence
if args.monitor:meta['monitor_sha256']=hashlib.sha256(args.monitor.resolve().read_bytes()).hexdigest()
out=W/(TAG+'.json')
def save():out.write_text(json.dumps(meta,indent=2)+'\n')
def props():
 with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/props') as r:return json.load(r).get('memory',{})
def request(prompt):
 body={'model':MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':1234,'max_tokens':32,'enable_mtp':False,'enable_pld':False,'stream':False}
 t=time.monotonic();wall_start=time.time()
 with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as r:response=json.load(r)
 return {'wall_s':time.monotonic()-t,'started_unix':wall_start,'ended_unix':time.time(),'response':response,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
if listening():raise RuntimeError('port occupied')
owner='ple-packed-ab-'+str(os.getpid())
subprocess.run([str(args.lock.resolve()),'acquire',owner],check=True)
try:
 for name,fused,packed in [('base',False,False),('current',True,False),('packed',True,True),('ahead',True,True),('combined',True,True),('group',True,False)]:
  if args.sequence and name!='current':continue
  if name not in os.environ.get('PLE_BENCH_ARMS','current,combined').split(','):continue
  env={k:v for k,v in os.environ.items() if not k.startswith(('MLX_SERVE_','QWEN4_'))};env.update(DYLD_LIBRARY_PATH=str(P/'lib/mlx/lib')+':'+str(P/'lib/llama/lib'),MLX_SERVE_PREFILL_CHUNK='8192',MLX_SERVE_PREFILL_TRACE='1',MLX_SERVE_QSA_PAIR=str(int(fused)),MLX_SERVE_HC_PREFILL=str(int(fused)),MLX_SERVE_GDN_PREFILL_FUSED=str(int(fused)),MLX_SERVE_PLE_PACKED=str(int(packed)),MLX_SERVE_PLE_AHEAD=str(int(name in ('ahead','combined'))),MLX_SERVE_PLE_TIMING='1',MLX_SERVE_MOE_PREFILL_GROUP=str(int(name in ('combined','group'))),QWEN4_PREFILL_CADENCE_TIMING='1')
  if os.environ.get('PREFILL_CAPTURE_ROUTES')=='1':env['QWEN4_MOE_CAPTURE_PATH']=str(W/'routes')
  if os.environ.get('PREFILL_MOE_LIVE_AB')=='1':env['QWEN4_MOE_LIVE_AB']='1'
  if args.sequence:env['QWEN4_PREFILL_ARM_SEQUENCE']=args.sequence
  env.pop('QWEN4_PROFILE_FWD',None)
  if os.environ.get('PLE_BENCH_CPU_TIMING')=='1':env['QWEN4_PROFILE_FWD']='1'
  logpath=W/(TAG+'-'+name+'.log')
  with logpath.open('w') as log:
   start=time.monotonic();server=subprocess.Popen([str(BIN),'serve','--host','127.0.0.1','--port',str(PORT),'--ctx-size','131072','--prefix-cache-entries','0','--prefill-chunk','8192','--kv-quant','off','--no-mtp','--no-pld','--no-drafter'],env=env,stdout=log,stderr=log)
   monitor=None;monitor_log=None
   try:
    while not listening():
     if server.poll() is not None:raise RuntimeError(logpath.read_text()[-4000:])
     if time.monotonic()-start>30:raise TimeoutError('bind')
     time.sleep(.2)
    ready=request('Capital of France? Answer with one word.')
    assert 'paris' in ready['response']['choices'][0]['message']['content'].lower(),ready
    if args.monitor:
     monitor_log=(W/(TAG+'-'+name+'-telemetry.jsonl')).open('w')
     monitor=subprocess.Popen([str(args.monitor.resolve()),'pipe','--interval','1000','--samples','0'],stdout=monitor_log,stderr=monitor_log)
    arm={'arm':name,'loaded_s':time.monotonic()-start,'memory_before':props(),'env':{k:v for k,v in env.items() if k.startswith(('MLX_SERVE_','QWEN4_'))},'requests':[]}
    records.append(arm);save();print(json.dumps({k:v for k,v in arm.items() if k!='requests'}),flush=True)
    cases=([('81492017',40000),('52839106',193500)] if os.environ.get('PLE_BENCH_LONG')=='1' else [('81492017',40000),('52839106',47000)])
    if args.sequence:cases=[('81492017',193500 if os.environ.get('PLE_BENCH_LONG')=='1' else 40000)]*len(args.sequence)
    for request_index,(nonce,size) in enumerate(cases[:1] if diagnostic and not args.trace else cases):
     prompt=f'Nonce {nonce}. Remember the passphrase MAGNOLIA-7731.\n'+(corpus*(16 if os.environ.get('PLE_BENCH_LONG')=='1' else 4))[:size]+'\nWhat was the passphrase? Answer with the passphrase only.'
     tracer=None;trace_log=None
     if args.trace and size==47000:
      trace_log=(W/(TAG+'-'+name+'-trace.log')).open('w')
      tracer=subprocess.Popen(['xcrun','xctrace','record','--template','Metal System Trace','--attach',str(server.pid),'--time-limit','12s','--output',str(W/(TAG+'-'+name+'.trace')),'--no-prompt'],stdout=trace_log,stderr=trace_log)
      time.sleep(2)
     offset=logpath.stat().st_size;r=request(prompt);time.sleep(.1)
     if tracer is not None:
      for attempt in range(4):
       try: tracer.wait(timeout=45);break
       except subprocess.TimeoutExpired: print('Waiting for trace finalization',flush=True)
      if tracer.poll() is None:tracer.terminate();tracer.wait(timeout=10)
      r['trace_exit_code']=tracer.returncode
      trace_log.close()
     emitted=logpath.read_bytes()[offset:].decode(errors='replace')
     if args.sequence:
      expected=int(args.sequence[request_index]);assert f'request={request_index} arm={expected}' in emitted,emitted
      r['experiment_arm']=expected
     rates=re.findall(r'prefill: ([0-9.]+) tok/s, decode: ([0-9.]+) tok/s',emitted)
     r.update(prefill_tps=float(rates[-1][0]) if rates else None,decode_tps=float(rates[-1][1]) if rates else None,memory_after=props(),server_log=emitted)
     arm['requests'].append(r);save()
     response=r['response'];usage=response['usage'];answer=response['choices'][0]['message']['content']
     print(json.dumps({'arm':name,'experiment_arm':r.get('experiment_arm'),'prefill_tps':r['prefill_tps'],'decode_tps':r['decode_tps'],'usage':usage,'answer':answer,'memory_after':r['memory_after']}),flush=True)
     assert usage.get('prompt_tokens_details',{}).get('cached_tokens',0)==0,usage
     assert 'MAGNOLIA-7731' in answer,response
    text=logpath.read_text()
    assert ('[ple-packed] engaged:' in text)==(any(c in args.sequence for c in '23') if args.sequence else packed)
    if name in ('ahead','combined'):assert '[ple-ahead] chunks=' in text and 'chunks=0/' not in text,text[-4000:]
    if name in ('combined','group'):assert '[moe-prefill-group] engaged:' in text
    if fused:assert '[qsa-pair] engaged:' in text and '[hc-prefill] engaged:' in text and '[gdn-prefill] engaged:' in text
   finally:
    if monitor is not None:
     monitor.terminate();monitor.wait(timeout=10);monitor_log.close()
    server.terminate()
    try:server.wait(timeout=20)
    except subprocess.TimeoutExpired:server.kill();server.wait()
    deadline=time.monotonic()+10
    while listening() and time.monotonic()<deadline:time.sleep(.2)
    if listening():raise RuntimeError('owned port did not close')
finally:subprocess.run([str(args.lock.resolve()),'release',owner],check=True)
