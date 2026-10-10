from conveyor import LineCounter,detect_parts
from app import synthetic
import numpy as np
def detection(x,y=80):return {'center':(x,y),'box':(int(x)-5,y-5,10,10),'area':100}
def test_crossing_counted_once():
    c=LineCounter(line_x=100)
    for x in [70,80,90,97,101,109,115,108,98,90,110]:c.update([detection(x)])
    assert c.total==1
def test_reverse_and_spawn_beyond_line_not_counted():
    c=LineCounter(line_x=100)
    for x in [130,120,110,100,90,80]:c.update([detection(x)])
    assert c.total==0
def test_brief_dropout_preserves_identity():
    c=LineCounter(line_x=100)
    for x in [70,80,89]:c.update([detection(x)])
    c.update([]);c.update([detection(111)]);assert c.total==1 and c.next_id==2
def test_dead_tracks_expire():
    c=LineCounter(max_missing=2);c.update([detection(20)])
    for _ in range(3):c.update([])
    assert not c.tracks
def test_background_has_no_detections():assert detect_parts(np.full((80,80,3),60,np.uint8))==[]
def test_unseen_synthetic_seed():
    frames,truth=synthetic(2026,count=6);c=LineCounter()
    for frame in frames:c.update(detect_parts(frame))
    assert c.total==truth==6
