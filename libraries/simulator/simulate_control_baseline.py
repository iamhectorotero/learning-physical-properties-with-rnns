'''Call javascript implementation of task from within python'''

import pyduktape
import numpy as np
import json
import random as rd
import math
import imp

#Read in mouse data
###################
# with open('../json/mouse_data.json') as data_file:    
#     mouse_data = json.load(data_file)

#Read in starting conditions
#############################
# with open('../json/starting_conditions.json') as data_file:    
#     starts = json.load(data_file)
with open('json/p10t4.json') as data_file:    
    trial = json.load(data_file)

#Create js environment
######################
context = pyduktape.DuktapeContext()

#Import js library
##################
# js_file = open("js/box2d.js")
# js = file.read(js_file)
context.eval_js_file("js/box2d.js")

#Load the js script
################
# js_file = open("../js/control_world.js")
# js = file.read(js_file)
context.eval_js_file("js/control_world.js")

#Run through a trial
######################################

rd.seed(2)#Set seed

#Choose a starting condition at random from participant data
# ix = rd.sample(range(1, len(starts['frame'])), 1)[0]
start = trial['start']
setup = trial['setup']
onset = trial['onset']
offset = trial['offset']

#Set starting conditions
cond = {'sls':[{'x':start['o1.x'], 'y':start['o1.y']}, {'x':start['o2.x'], 'y':start['o2.y']},
       {'x':start['o3.x'], 'y':start['o3.y']}, {'x':start['o4.x'], 'y':start['o4.y']}],
        
    'svs':[{'x':start['o1.vx'], 'y':start['o1.vy']}, {'x':start['o2.vx'], 'y':start['o2.vy']},
        {'x':start['o3.vx'], 'y':start['o3.vy']}, {'x':start['o4.vx'], 'y':start['o4.vy']}]
    }

cond['lf'] = [[0.0,float(setup['lf1']), float(setup['lf2']), float(setup['lf3'])],
        [0.0, 0.0, float(setup['lf4']),float(setup['lf5'])],
        [0.0, 0.0, 0.0, float(setup['lf6'])],
        [0.0, 0.0, 0.0, 0.0]]

if setup['mass']=='A':
    cond['mass'] = [2,1,1,1]
elif setup['mass']=='B':
    cond['mass'] = [1,2,1,1]
else:
    cond['mass'] = [1,1,1,1]

path = {'x':np.repeat(1, onset['frame'][0]).tolist(),
'y':np.repeat(1, onset['frame'][0]).tolist(),
'obj':np.repeat(0, onset['frame'][0]).tolist()}
print(path)

cond['timeout']=len(path['x'])

#Simulate in javascript
########################
context.set_globals(cond=cond)
context.set_globals(control_path=path)

# print(path["obj"])
#Run the simulation
###################
data = context.eval_js("Run();")
data = json.loads(data) #Convert to python object

#Check something happened
#########################
print('data', data['physics'])
#data['physics']['o1']['y'][0:100]

#Evaluate TODO
# data['physics']['o1']['x'][0:100]
# with open('../../R/data/replay_files_exp4/ppt_10_uid_A14ADQ7RUN6TDY.json') as data_file:    
# ppt_data = json.load(data_file)

# ppt_data[4]['o1x'[0:100]]


#Save data
##########
# with open('../json/test_sim.json', 'w') as fp:
# json.dump(data, fp, sort_keys=True, indent=4)


#Make a movie
############

# imp.load_source("make_movies_simulation.py", "../make_movies/")
# import ../make_movies/make_movies_simulation

#Plot things out
###############
# import gizeh
# surface = gizeh.Surface(width=320, height=260) # in pixels
# circle = gizeh.circle(r=30, xy= [40,40], fill=(1,0,0))
# circle.draw(surface) # draw the circle on the surface
# surface.write_to_png("circle.png") # export the surface as a PNG
