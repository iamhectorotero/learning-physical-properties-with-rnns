import pandas as pd
import numpy as np
import gizeh as gz
import math
import gizeh as gz
import moviepy.editor as mpy

from .dataset import read_dataset

def make_frame_curried(this_data):
    def make_frame(t):
        labels = ['A', 'B', '', '']
        centers = np.array(['o1','o2','o3','o4'])
        colors = [(1,0,0),(0,1,0),(0,0,1),(0,0,1)]
        RATIO = 100
        RAD = 25
        W = 600
        H = 400
        # H_outer = 500
        N_OBJ = 4

        frame = int(math.floor(t*60))
        # Essentially pauses the action if there are no more frames and but more clip duration
        # if frame >= len(this_data["co"]):
        #    frame = len(this_data["co"])-1

        # White background
        surface = gz.Surface(W,H, bg_color=(1,1,1))

        # Walls
        wt = gz.rectangle(lx=W, ly=20, xy=(W/2,10), fill=(0,0,0))#, angle=Pi/8
        wb = gz.rectangle(lx=W, ly=20, xy=(W/2,H-10), fill=(0,0,0))
        wl = gz.rectangle(lx=20, ly=H, xy=(10,H/2), fill=(0,0,0))
        wr = gz.rectangle(lx=20, ly=H, xy=(W-10,H/2), fill=(0,0,0))
        wt.draw(surface)
        wb.draw(surface)
        wl.draw(surface)
        wr.draw(surface)

        # Pucks
        for label, color, center in zip(labels, colors, centers):

            xy = np.array([this_data[center+'.x'].iloc[frame]*RATIO, this_data[center+'.y'].iloc[frame]*RATIO])

            ball = gz.circle(r=RAD, fill=color).translate(xy)
            ball.draw(surface)

            # Letters
            text = gz.text(label, fontfamily="Helvetica",  fontsize=25, fontweight='bold', fill=(0,0,0), xy=xy) #, angle=Pi/12
            text.draw(surface)

        # Mouse cursor
        if 'mouse' in data.columns:
            cursor_xy = np.array([this_data['mouse.x'].iloc[frame]*RATIO, this_data['mouse.y'].iloc[frame]*RATIO])
        else:
            cursor_xy = np.array([0, 0])

        cursor = gz.text('+', fontfamily="Helvetica",  fontsize=25, fill=(0,0,0), xy=cursor_xy) #, angle=Pi/12
        cursor.draw(surface)

        # Control
        if "co" in this_data.columns and this_data['co'].iloc[frame] != 0:
            if this_data['co'][frame]==1:
                xy = np.array([this_data['o1.x'].iloc[frame]*RATIO, this_data['o1.y'].iloc[frame]*RATIO])
            elif this_data['co'][frame]==2:
                xy = np.array([this_data['o2.x'].iloc[frame]*RATIO, this_data['o2.y'].iloc[frame]*RATIO])
            elif this_data['co'][frame]==3:
                xy = np.array([this_data['o3.x'].iloc[frame]*RATIO, this_data['o3.y'].iloc[frame]*RATIO])
            elif this_data['co'][frame]==4:
                xy = np.array([this_data['o4.x'].iloc[frame]*RATIO, this_data['o4.y'].iloc[frame]*RATIO])

            # control_border = gz.arc(r=RAD, a1=0, a2=np.pi, fill=(0,0,0)).translate(xy)
            control_border = gz.circle(r=RAD, stroke_width=2).translate(xy)
            control_border.draw(surface)

        return surface.get_npimage()

    return make_frame


if __name__ == "__main__":

    random_trial = np.random.randint(0, 1600)
    random_trial = 1
    data = pd.read_hdf("cond_1.h5", key="trial_"+str(random_trial))

    print(data.head())
    make_frame = make_frame_curried(data)
    duration = data.shape[0]/60
    clip = mpy.VideoClip(make_frame, duration=duration)

    # Create the filename (adding 0s to ensure things are in a nice alphabetical order now)
    writename = "trial_"+str(random_trial)+".mp4"

    # Write the clip to file
    clip.write_videofile(writename, fps=24)

