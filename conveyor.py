"""Colour segmentation, centroid association and directional line counting."""
from dataclasses import dataclass
import cv2
import numpy as np

def detect_parts(frame, min_area=150, saturation=65):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,saturation,65]),np.array([179,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    return [{'center':tuple(map(float,centers[i])),'box':tuple(map(int,stats[i,:4])),'area':int(stats[i,4])} for i in range(1,n) if stats[i,4]>=min_area]

@dataclass
class Track:
    identifier:int
    center:tuple
    box:tuple
    hits:int=1
    missing:int=0
    side:int=0
    counted:bool=False

class LineCounter:
    def __init__(self,line_x=320,max_distance=55,max_missing=8,hysteresis=8):
        self.line_x=line_x;self.max_distance=max_distance;self.max_missing=max_missing;self.hysteresis=hysteresis
        self.tracks={};self.next_id=1;self.total=0;self.events=[];self.frame=0
    def side(self,x):
        return -1 if x<self.line_x-self.hysteresis else (1 if x>self.line_x+self.hysteresis else 0)
    def update(self,detections):
        self.frame+=1
        pairs=sorted((float(np.linalg.norm(np.array(t.center)-d['center'])),tid,j) for tid,t in self.tracks.items() for j,d in enumerate(detections))
        used_t,used_d=set(),set()
        for distance,tid,j in pairs:
            if distance>self.max_distance or tid in used_t or j in used_d:continue
            t=self.tracks[tid];d=detections[j];new_side=self.side(d['center'][0])
            if not t.counted and t.side==-1 and new_side==1 and t.hits>=3:
                self.total+=1;t.counted=True;self.events.append({'frame':self.frame,'track_id':tid,'direction':'left-to-right'})
            if new_side:t.side=new_side
            t.center=d['center'];t.box=d['box'];t.hits+=1;t.missing=0;used_t.add(tid);used_d.add(j)
        for tid,t in list(self.tracks.items()):
            if tid not in used_t:
                t.missing+=1
                if t.missing>self.max_missing:del self.tracks[tid]
        for j,d in enumerate(detections):
            if j not in used_d:
                tid=self.next_id;self.next_id+=1
                self.tracks[tid]=Track(tid,d['center'],d['box'],side=self.side(d['center'][0]))
        return [t for t in self.tracks.values() if t.missing==0]

def annotate(frame,tracks,counter,label='CONVEYOR VISION'):
    out=frame.copy();cv2.line(out,(counter.line_x,48),(counter.line_x,out.shape[0]-30),(69,176,239),2)
    for t in tracks:
        x,y,w,h=t.box;color=(110,225,165) if t.counted else (229,207,132)
        cv2.rectangle(out,(x,y),(x+w,y+h),color,2)
        cv2.putText(out,f'ID {t.identifier}',(x,max(56,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(238,242,230),1)
    cv2.rectangle(out,(0,0),(out.shape[1],44),(26,34,28),-1)
    cv2.putText(out,f'{label} | COUNT {counter.total}',(17,29),cv2.FONT_HERSHEY_SIMPLEX,.60,(216,233,214),1)
    return out
