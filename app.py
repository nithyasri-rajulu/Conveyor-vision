import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from conveyor import detect_parts,LineCounter,annotate

def synthetic(seed=7,count=12,frames=260):
    rng=np.random.default_rng(seed)
    parts=[{'x':-65-(i//3)*150,'y':95+(i%3)*119+int(rng.integers(-8,9)),'v':float(rng.uniform(4.1,4.7)),'r':int(rng.integers(14,22)),'color':[(85,168,231),(211,151,80),(93,173,91)][i%3],'circle':i%2==0} for i in range(count)]
    expected=sum(p['x']+p['v']*(frames-1)>328 for p in parts)
    def images():
        for f in range(frames):
            frame=np.full((420,640,3),(48,54,50),np.uint8)
            for y in [53,155,274,380]:cv2.line(frame,(0,y),(639,y),(73,80,73),1)
            for x in range(-30,670,45):cv2.line(frame,(int((x+f*2)%690-25),55),(int((x+f*2)%690-25),380),(61,68,62),1)
            for p in parts:
                x=int(p['x']+p['v']*f);y=p['y'];r=p['r']
                if p['circle']:cv2.circle(frame,(x,y),r,p['color'],-1)
                else:cv2.rectangle(frame,(x-r,y-r),(x+r,y+r),p['color'],-1)
            frame=np.clip(frame.astype(float)*(1+.08*np.sin(f/30))+rng.normal(0,1.5,frame.shape),0,255).astype(np.uint8)
            yield frame
    return images(),expected

def run(frames,output=None,save=False,line_x=320):
    tracker=LineCounter(line_x);writer=None;gif=[];ms=[]
    try:
        for i,frame in enumerate(frames):
            start=time.perf_counter();tracks=tracker.update(detect_parts(frame));ms.append((time.perf_counter()-start)*1000)
            image=annotate(frame,tracks,tracker,'SYNTHETIC BENCHMARK' if save else 'CONVEYOR VISION')
            if save:
                if writer is None:
                    writer=cv2.VideoWriter(str(output/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(frame.shape[1],frame.shape[0]))
                    if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
                writer.write(image)
                if i%5==0:gif.append(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
                if i==145:cv2.imwrite(str(output/'preview.png'),image)
        if not ms:raise ValueError('No frames were decoded')
        if save:
            gif[0].save(output/'demo.gif',save_all=True,append_images=gif[1:],duration=200,loop=0)
        return {'count':tracker.total,'frames':len(ms),'mean_processing_ms':round(float(np.mean(ms)),3),'events':tracker.events}
    finally:
        if writer:writer.release()

def video_frames(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot open video: {path}')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            yield frame
    finally:cap.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');p.add_argument('--video',type=Path);p.add_argument('--output',type=Path,default=Path('outputs'));p.add_argument('--line-x',type=int,default=320);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.demo:
        runs=[]
        for seed in [7,19,31,43,59]:
            frames,truth=synthetic(seed);r=run(frames,a.output,seed==7);r.update(seed=seed,expected=truth,absolute_count_error=abs(r['count']-truth));runs.append(r)
        result={'dataset':'five synthetic separated-lane clips; no real factory footage','seeds':[7,19,31,43,59],'total_expected':sum(r['expected'] for r in runs),'total_counted':sum(r['count'] for r in runs),'mean_absolute_count_error':float(np.mean([r['absolute_count_error'] for r in runs])),'runs':runs}
    elif a.video:result=run(video_frames(a.video),line_x=a.line_x)
    else:p.error('Choose --demo or --video FILE')
    (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['runs','events']},indent=2))
