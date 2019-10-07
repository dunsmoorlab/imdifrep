'''
Day1 ExPress
Dual CS+ conditioning with Suppression facilitiated extinction
'''
import argparse
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
from psychopy import iohub, visual, core, gui, data, monitors, tools, parallel
from psychopy import event as Event
import pickle
import time

#########################
##  utility functions  ##
#########################
def check4quit():
    for kp in keyboard.getPresses(keys=QUIT_KEY,mods=[QUIT_MODIFIER]):
        io.quit();win.close();sys.exit()

def slackit(msg):
    print(msg)
    if SLACK:
        payload = dict(text=msg,channel=SLACK['channel'],username=SLACK['botname'],icon_emoji=SLACK['emoji'])
        try: requests.post(json=payload,url=SLACK['url'])
        except ConnectionError: print('Slack messaging failed--no internet connection.')

def wait_for_scanner():
    win.flip()
    Event.clearEvents()
    while True:
        check4quit()
        if SCAN_TRIGGER in keyboard.state: break
        # for kp in keyboard.getPresses(keys=SCAN_TRIGGER): break

# for simulated keypresses/responses
class sim_kp(object):
    def __init__(self):
        self.char = np.random.choice(RESP_KEYS)
        self.time = np.random.normal(1.2,.1) + core.getTime()

def prep_for_response():
    Event.clearEvents()
    return core.getTime(), None

def collect_response():
    for kp in keyboard.getPresses(keys=RESP_KEYS):
        return kp
    if SUBJ == 'sim':
        return sim_kp()

def stamp_next_flip(on_or_offset,stamp_idx):
    stamp_df.loc[stamp_idx,'true_{:s}'.format(on_or_offset)] = clock.getTime()

def stamp_onoffset(on_or_offset,stamp_idx,BIO=False,SHOCK=False):
    win.callOnFlip(stamp_next_flip,on_or_offset,stamp_idx)

    if BIO and 'stim' in stamp_idx[1]:
        if 'on' in on_or_offset:
            port.setPin(4,1)
        elif 'off' in on_or_offset:
            port.setPin(4,0)


SCREENS = {
    'skyrawide':   dict(distance_cm=136,width_cm=85.7,pixel_dims=[1920,1080]),
    'skyrashit':   dict(distance_cm=136,width_cm=38.0,pixel_dims=[1024, 768]),
    'skyrasmall':  dict(distance_cm=136,width_cm=45.0,pixel_dims=[1024, 768]),
    'gus':         dict(distance_cm=67,width_cm=59.3,pixel_dims=[2560,1440]),
    'home':        dict(distance_cm=100,width_cm=68.47,pixel_dims=[3840, 2160]),
    'factory'   :  dict(distance_cm=136,width_cm=85.7,pixel_dims=[1920,1080]) 
}


parser = argparse.ArgumentParser(description='day1 conditioning & extinction')
parser.add_argument('-s','--subj',default='s666',type=str,help='subject #')
parser.add_argument('-c','--scrn',default='factory',type=str,choices=list(SCREENS.keys()),help='screen used for display')
parser.add_argument('-o','--order',default=None,type=str,choices=['A','T'],help='which category is the suppresed category.')
parser.add_argument('-p','--phase',default=None,type=str,help='start from this phase, probably for dev only')

args = parser.parse_args()

#=================================================
# Set some stimulus parameters and other constants
#=================================================
PROJ_NAME = 'express'
SUBJ = args.subj
SCRN = args.scrn
ORDER = args.order
DEV = SUBJ in ['s666','sim']
SCAN = 'skyra' in SCRN
BIO = 'factory' in SCRN
SESS = 'memory'
DATE = data.getDateStr()

home = os.path.expanduser('~')

#i/o
if 'factory' in SCRN:
    # data_dir = 'C://Users//dunsmoorlab//Desktop//Experiments//{:s}//{:s}'.format(PROJ_NAME,SUBJ)
    data_dir = os.path.join(home,'Desktop','Experiments',PROJ_NAME,SUBJ)
else:
    data_dir = os.path.join(home,'Db_lpl','STUDY','ExPress')

#read in the design file with everything on it
design = pd.read_csv('express_design.csv')

#initialize the parallel port if in the testing room
if BIO:
    port = parallel.ParallelPort(address=0xDFF8)
    port.setData(0)

#initialize some vairables
N_RUNS      = 4 # number of runs
N_TRIALS    = 60 # total per run
STIM_SECS   = 6 # day1 encoding duration
ITI_SECS    = 1
phase_names = ['mem1','mem2','mem3','mem4'] # the names of each phase for memory_runs

# there are 4 groups of animals/tools/food that are always seen together,
# these lines randomize which group of images go to which phase
blocks = [1,2,3,4]
np.random.shuffle(blocks)

STIMLIST = {} # create the stimlists for day1
for run, phase in enumerate(phase_names): # one for each run


    old_animals = np.array(design.day1_animals[design.day2_stim_block == blocks[run]]) #collect a block of animals
    old_tools = np.array(design.day1_tools[design.day2_stim_block == blocks[run]]) #and a block of tools
    old_food = np.array(design.day1_food[design.day2_stim_block == blocks[run]]) #and a block of food

    new_animals = np.array(design.foil_animals[design.day2_foil_block == blocks[run]])
    new_tools = np.array(design.foil_tools[design.day2_foil_block == blocks[run]])
    new_food = np.array(design.foil_food[design.day2_foil_block == blocks[run]])
    # and randomize them
    np.random.shuffle(old_animals)
    np.random.shuffle(old_tools)
    np.random.shuffle(old_food)

    np.random.shuffle(new_animals)
    np.random.shuffle(new_tools)
    np.random.shuffle(new_food)
    # determine which category is the CS+E,CS+,and CS-
    if ORDER == 'A': 
        cspe_cat = 'A'
        csp_cat = 'T'
    
    elif ORDER == 'T':
        cspe_cat = 'T'
        csp_cat = 'A'
    
    old_csmap = {'A':old_animals,'T':old_tools,'F':old_food}
    old_CS = {'CS+E': old_csmap[cspe_cat],
          'CS+' : old_csmap[csp_cat],
          'CS-' : old_csmap['F']}
    
    new_csmap = {'A':new_animals,'T':new_tools,'F':new_food}
    new_CS = {'CS+E': new_csmap[cspe_cat],
          'CS+' : new_csmap[csp_cat],
          'CS-' : new_csmap['F']}


    csorder = np.array(design['%s_proc'%(phase)].dropna())
    oldnew = np.array(design['%s_oldnew'%(phase)].dropna())

    run_stims = np.empty(len(csorder),dtype=object) #empty array where stims will go
    for condition in old_CS.keys(): #for each CS+/CS-
        cs_map = np.where(csorder == condition)[0] #find where they are in the order of the phase
        run_stims[cs_map[np.where(oldnew[cs_map] == 'old')[0]]] = old_CS[condition] #and put them in place
        run_stims[cs_map[np.where(oldnew[cs_map] == 'new')[0]]] = new_CS[condition] #and put them in place
    STIMLIST[phase_names[run]] = run_stims #store them for each phase

#set up the dataframe with all the timings
EVENTS = OrderedDict()
for phase in phase_names:
    itis = np.tile([.75,.75],int(N_TRIALS/2))
    np.random.shuffle(itis)

    EVENTS[phase] = OrderedDict(
    [ x for y in 
        [ (('stim{:d}'.format(i+1),STIM_SECS),('iti{:d}'.format(i+1),itis[i])) for i in range(N_TRIALS) ] for x in y ])

    #add the initial iti to every run
    EVENTS[phase]['iti0'] = 2
    EVENTS[phase].move_to_end('iti0',last=False)

# screen
FRAME_RATE = 60. # flips per second
BKGRND_COLOR = 'black'

# keyboard
QUIT_KEY = '='
QUIT_MODIFIER = 'lshift'
SCAN_TRIGGER = '5'
BREAK_KEY = 'space'
RESP_KEYS = ['1','2','3','4']


# stims
FIX_RAD      = 0.25  # fixation dot
TARG_RAD     = 5     # memory image targets
BREAK_COLOR  = 'white'


#################################
##  initialize file structure  ##
#################################

data_basename  = 'behavior_{:s}_data.csv'.format(SESS)
stamp_basename = 'behavior_{:s}_stamps.csv'.format(SESS)
data_fname  = os.path.join(home,data_dir,data_basename)
stamp_fname = os.path.join(home,data_dir,stamp_basename)

# make sure we wont overwrite any files
load = False
load_phase = False
if not DEV and os.path.isfile(data_fname):
    print('WARNING: Already a file with this subj and session!')
    move_forward = False
    while move_forward not in ['y','n']:
        move_forward = input('Continue anyways? (y or n)\n')
    if move_forward == 'n':
        sys.exit()
    elif move_forward == 'y':
        load = False
        print('Would you like to load previous data and continue experiment?')
        while load not in ['y', 'n']:
            load = input('(y or n)\n')
        if load == 'n':
            sys.exit()
        elif load == 'y':
            print('Enter the phase you would like to load:')
            load_phase = False
            while load_phase not in phase_names:
                load_phase = input('(%s)\n'%(phase_names))

if load == 'y' and load_phase in phase_names:
    df = pd.read_csv(data_fname)
    stamp_df = pd.read_csv(stamp_fname)
else:
    # create a new directory if first time running subj
    if not os.path.isdir(os.path.dirname(data_fname)):
        os.mkdir(os.path.dirname(data_fname))

    ########################
    ##  build dataframes  ##
    ########################

    idx = pd.IndexSlice

    # main df
    df_cols = [ 'subj',
                'sess',
                'imgFname',
                'resp',
                'rt',
                'condition',
                'oldnew']

    phases1_ = np.repeat([phase_names],np.tile(N_TRIALS,N_RUNS))
    events1_ = np.tile(np.array(range(1,N_TRIALS+1)),N_RUNS)
    tupes1 = list(zip(phases1_,events1_))

    phases2_ = np.repeat([phase_names],np.tile(121,N_RUNS)) #these values have to be set manually bc of the pre and post cues during extinction
    events2_ = np.concatenate((list(EVENTS['mem1'].keys()),list(EVENTS['mem2'].keys()),list(EVENTS['mem3'].keys()),list(EVENTS['mem4'].keys())))
    tupes2 = list(zip(phases2_,events2_))


    df_indx = pd.MultiIndex.from_tuples(tupes1,names=['phase','trial'])
    df = pd.DataFrame(columns=df_cols,index=df_indx,dtype=float)
    df['subj'] = SUBJ
    df['sess'] = SESS
    df['date'] = DATE
    df['day2_block'] = 0
    # stamp df
    stamp_cols = ['planned_onset','planned_offset','true_onset','true_offset']
    stamp_indx = pd.MultiIndex.from_tuples(tupes2,names=['phase','event'])
    stamp_df = pd.DataFrame(columns=stamp_cols,index=stamp_indx)

    for run, phase in enumerate(phase_names):
        running_time = 0
        for event, length in list(EVENTS[phase].items()):
            stamp_df.loc[(phase,event),'planned_onset'] = running_time
            stamp_df.loc[(phase,event),'planned_offset'] = running_time + length
            running_time += length
        df['imgFname'].loc[phase] = STIMLIST[phase]
        df['condition'].loc[phase] = design['%s_proc'%(phase)].dropna().values
        df['day2_block'].loc[phase] = blocks[run]

        df['oldnew'].loc[phase] = design['%s_oldnew'%(phase)].dropna().values
    # sort the both indices of both dataframes for slightly faster indexing
    # # (will reset before exporting for easier read)
    # for axis in ['index','columns']:
    #     df.sort_index(axis=axis,inplace=True)
    #     stamp_df.sort_index(axis=axis,inplace=True)

###########################
##  open/setup psychopy  ##
###########################

# keyboard and clock
io = iohub.launchHubServer(); io.clearEvents('all')
keyboard = io.devices.keyboard

clock = core.Clock()

# screen
mon = monitors.Monitor('testMonitor')
mon.setDistance(SCREENS[SCRN]['distance_cm'])
mon.setWidth(SCREENS[SCRN]['width_cm'])
mon.setSizePix(SCREENS[SCRN]['pixel_dims'])
win = visual.Window(monitor=mon,units='deg',fullscr=True,color=BKGRND_COLOR)
mouse = Event.Mouse(win=win,visible=False)
# check to make sure frame rate is as expected, since it's used above for flip buffer
actual_frame_rate = win.getActualFrameRate() 
if not (FRAME_RATE-2 < actual_frame_rate < FRAME_RATE+2):
    raise Warning('Frame rate ({}) is not as expected ({}).'.format(actual_frame_rate,FRAME_RATE))
flip_buffer = 1./FRAME_RATE


FIX_COLOR    = 'black'
SUPP_COLOR   = 'blue'
#initialize the stimuli
fixStim   = visual.TextStim(win,'+',pos=[0,0],color=FIX_COLOR,height=2)
whiteBk   = visual.ImageStim(win,'whitebackground.jpg',pos=[0,0],size=[TARG_RAD*2,TARG_RAD*2])
imgStim   = visual.ImageStim(win,pos=[0,0],size=[TARG_RAD*2,TARG_RAD*2])
txtStim   = visual.TextStim(win,pos=[0,0],height=.5) #,wrapWidth=req_dva_size)

#day2 response options
day2_DN   = visual.TextStim(win,'Definitely\n     New\n       1',pos=[-5,-6.5],height=.5)
day2_MN   = visual.TextStim(win,'Maybe\n  New\n     2',pos=[-1.5,-6.5],height=.5)
day2_MO   = visual.TextStim(win,'Maybe\n   Old\n     3',pos=[1.5,-6.5],height=.5)
day2_DO   = visual.TextStim(win,'Definitely\n     Old\n       4',pos=[5,-6.5],height=.5)

cf = {  '1': day2_DN, 
        '2': day2_MN,
        '3': day2_MO,
        '4': day2_DO}

#######################
##  event functions  ##
#######################

def instructions(phase,run):
    if phase in phase_names:
        msg = 'Press SPACE to begin memory run %s out of 4'%(run+1)
        txtStim.text = msg
        Event.clearEvents()
        txtStim.draw()
        win.flip()
        Event.waitKeys(keyList=['space'])


def show_event(event,phase,BIO):
    until = stamp_df.loc[(phase,event),'planned_offset']
    start = clock.getTime()

    if 'stim' in event:
        #set image
        tnum = int(''.join([ x for x in event if x.isdigit() ]))
        img_fname = df.loc[(phase,tnum),'imgFname']
        imgStim.setImage(img_fname)
        
        next_iti = tnum
        next_stim = int(tnum + 1)
        this_stim = next_iti

        stim_list = [imgStim,day2_DN,day2_MN,day2_MO,day2_DO]

    elif 'iti' in event:
        stim_list = [whiteBk,fixStim]
    else:
        stim_list = []

    stamp_onoffset('onset',stamp_idx=(phase,event),BIO=BIO,SHOCK=False)
    if 'stim' in event:
    	t0, kp = prep_for_response()
    
    while clock.getTime() < until:
        for stim in stim_list:
            stim.draw()
        win.flip(); check4quit()
        if 'stim' in event:
            if kp is None:
                kp = collect_response()
                if kp is not None:
                    rt = kp.time - t0
                    savevals = [kp.char,rt]
                    if kp.char in RESP_KEYS:
                        cf[kp.char].color = 'orange'
                        until = start + rt + .3
                        stamp_df.loc[(phase,event)].planned_offset = until
                        stamp_df.loc[(phase,'iti%s'%(next_iti))].planned_onset = until
                        stamp_df.loc[(phase,'iti%s'%(next_iti))].planned_offset = until + ITI_SECS

                        if next_stim <= N_TRIALS:
                            stamp_df.loc[(phase,'stim%s'%(next_stim))].planned_onset = until + ITI_SECS


    stamp_onoffset('offset',stamp_idx=(phase,event),BIO=BIO)

    if 'stim' in event:
        savecols = ['resp','rt']
        if kp is None:
            savevals = [0,0]
        else:
            for k in cf.keys():
                cf[k].color = 'white'

        df.loc[(phase,tnum),savecols] = savevals

######################
##  run experiment  ##
######################
#this paragraph checks if we should start the experiment from a specific phase
if args.phase is not None: load_phase = args.phase

if load_phase in phase_names:
    where = np.array([phase == load_phase for phase in phase_names])
    for i, here in enumerate(where):
        if here: where[i:] = True
    run_phases = np.array(phase_names)[where]
elif load_phase not in phase_names: run_phases = phase_names

for r, phase in enumerate(run_phases):
    # break message
    # if not DEV:
    instructions(phase,r)
    # wait for scanner
    if SCAN:
        wait_for_scanner()
    clock.reset()
    for event in EVENTS[phase]:
        show_event(event,phase,BIO)
    win.flip() # to catch timestamp of final iti of run

    # export with more readable indices
    df.to_csv(data_fname,na_rep=np.nan)
    stamp_df.reindex(columns=stamp_cols,labels=stamp_indx).to_csv(stamp_fname,na_rep=np.nan)



txtStim.text = ('You are done with the experiment for today.')
txtStim.draw()
win.flip()
keyboard.waitForPresses(keys=QUIT_KEY,mods=(QUIT_MODIFIER,),maxWait=120)
win.close()