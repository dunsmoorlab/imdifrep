'''
Day1 ExPress
Dual CS+ conditioning with Suppression facilitiated extinction
In [12]: for cat in CATEGORIES:
    ...:     for subcat in CATEGORIES[cat]:
    ...:         df[cat+'_'+subcat] = [stim for stim in os.listdir(os.path.
    ...: join(data_dir,'exp_stims',cat,subcat))]
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


SCREENS = {
    'gus':         dict(distance_cm=67,width_cm=59.3,pixel_dims=[2560,1440]),
    'home':        dict(distance_cm=100,width_cm=68.47,pixel_dims=[3840, 2160]),
    'factory'   :  dict(distance_cm=136,width_cm=85.7,pixel_dims=[1920,1080])
    'lewpea':      dict(distance_cm= 60,width_cm=47.3,pixel_dims=[1920,1080]),
}

parser = argparse.ArgumentParser(description='directed forgetting & immediate recognition')
parser.add_argument('-s','--subj',default='s666',type=str,help='subject #')
parser.add_argument('-c','--scrn',default='lewpea',type=str,choices=list(SCREENS.keys()),help='screen used for display')
parser.add_argument('-o','--order',default=None,type=str,choices=['A','T'],help='which category is paired with forget cues')
parser.add_argument('-p','--phase',default=None,type=str,help='start from this phase, indented for dev only')

args = parser.parse_args()

#=================================================
# Set some stimulus parameters and other constants
#=================================================
PROJ_NAME = 'imdifrep'
SUBJ = args.subj
SCRN = args.scrn
ORDER = args.order
DEV = SUBJ in ['s666','sim','s777']
DATE = data.getDateStr()

home = os.getcwd()

#i/o
data_dir = os.getcwd()
assert os.path.split(data_dir)[-1] == PROJ_NAME #make sure we are in an imdifrep directory

#read in the design file with everything on it
design = pd.read_csv('imdifrep_design.csv')

#initialize the parallel port if in the testing room
if BIO:
    port = parallel.ParallelPort(address=0xDFF8)
    port.setData(0)

#initialize some vairables
N_RUNS               = 2 # number of runs
N_SUBCAT_STIMS       = 33 #number of subcat stims in encoding
N_CAT_STIMS          = int(N_SUBCAT_STIMS * 2) #number of stimuli per category used for encoding
N_ENCODING_TRIALS    = int(N_CAT_STIMS * 3) # total for encoding
N_MEM_TRIALS         = int(N_ENCODING_TRIALS*2) # double the amount of foils

STIM_SECS            = 3 # day1 encoding duration

phase_names = ['encoding','retrieval'] # the names of each phase

categories = {'animal':['land','nonland'],
              'tool'  :['manual','power'],
              'food'  :['cooked','uncooked']}

#decide which stims will be encoding and which will be foils
oldnew = np.random.permutation(np.repeat([phase_names],N_SUBCAT_STIMS))
old_loc = np.random.permutation(np.where(oldnew == 'encoding')[0])
new_loc = np.random.permutation(np.where(oldnew == 'retrieval')[0])

stims = pd.read_csv('stim_df.csv') #read in the list of all stims
encoding_stims = stims.loc[old_loc] #grab encoding
retrieval_stims = stims.loc[new_loc] #grab foils

#Make the encoding order, pseudoblocked...
#so that every group of 6 trials has equal # of each condition
conditions = ['F','R','X']
con_reps = 2
pblock = int(len(conditions) * con_reps)
N_ENC_BLOCKS = int(N_ENCODING_TRIALS/pblock)

encoding_order = np.zeros(( N_ENC_BLOCKS,pblock ), dtype=str)
for i in range(N_ENC_BLOCKS): encoding_order[i] = np.random.permutation(np.repeat(conditions,con_reps))
encoding_order = encoding_order.flatten()

X_cat = 'food' #food is always view
if ORDER == 'A': #given order determines the forget category
    F_cat = 'animal' 
    R_cat = 'tool'
elif ORDER == 'T':
    F_cat = 'tool'
    R_cat = 'animal'

conmap = {'F':F_cat,
           'R':R_cat,
           'X':X_cat}

#next need to grab the different sub-cats of stims, also in blocks of 6
#I guess techincally I should do it once for each subject but they switch so much between sitmuli i really don't think it would be possible for anyone to notice the difference...
N_SUBCAT_BLOCKS = int(N_CAT_STIMS/pblock)
subcat_order = np.zeros(( N_SUBCAT_BLOCKS, pblock),dtype=int)
for i in range(N_SUBCAT_BLOCKS): subcat_order[i] = np.random.permutation(np.repeat([0,1],3))
subcat_order = subcat_order.flatten()

conmap[eo]+'_'+categories[conmap[eo]][sco]

encoding_stimlist = np.zeros(N_ENCODING_TRIALS, dtype=object)

for cat in categories:
    cat_hold = np.zeros(N_CAT_STIMS,dtype=object)
    for i, subcat in enumerate(categories[cat]):
        column = cat+'_'+subcat
        cat_hold[np.where(subcat_order == i)] = encoding_stims[column].values
        #this is where i got on monday

STIMLIST = {} # create the stimlists for day1
for run, phase in enumerate(phase_names): # one for each run

    csorder = np.array(design['%s_proc'%(phase)].dropna())
    
    animals = np.array(design.day1_animals[design.day1_block == blocks[run]]) #collect a block of animals
    tools = np.array(design.day1_tools[design.day1_block == blocks[run]]) #and a block of tools
    food = np.array(design.day1_food[design.day1_block == blocks[run]]) #and a block of food

    # and randomize them
    np.random.shuffle(animals)
    np.random.shuffle(tools)
    np.random.shuffle(food)

    
    csmap = {'A':animals,'T':tools,'F':food}
    CS = {'CS+E': csmap[cspe_cat],
          'CS+' : csmap[csp_cat],
          'CS-' : csmap['F']}

    run_stims = np.empty(len(csorder),dtype=object) #empty array where stims will go
    for condition in CS.keys(): #for each CS+/CS-
        cs_map = np.where(csorder == condition)[0] #find where they are in the order of the phase
        run_stims[cs_map] = CS[condition] #and put them in place
    
    STIMLIST[phase_names[run]] = run_stims #store them for each phase

#set up the dataframe with all the timings
EVENTS = OrderedDict()
for phase in phase_names:
    itis = np.tile([7.5,9.5],int(N_TRIALS/2))
    np.random.shuffle(itis)
    if 'fear' in phase:

        EVENTS[phase] = OrderedDict(
        [ x for y in 
            [ (('stim{:d}'.format(i+1),STIM_SECS),('iti{:d}'.format(i+1),itis[i])) for i in range(N_TRIALS) ] for x in y ])
    elif 'ext' in phase:
        #go trial by trial and add the pre and post cue if CS+E
        EVENTS[phase] = OrderedDict()
        proc = design['%s_proc'%(phase)].dropna()
    
        for i, trial in enumerate(proc):
            if 'E' not in trial:
                EVENTS[phase]['stim{:d}'.format(i+1)] = STIM_SECS
                
                if i < N_TRIALS - 1 and proc[i+1] == 'CS+E':
                    EVENTS[phase]['iti{:d}'.format(i+1)] = itis[i] - 2
                else:
                    EVENTS[phase]['iti{:d}'.format(i+1)] = itis[i]
            
            elif 'E' in trial:
                EVENTS[phase]['precue{:d}'.format(i+1)] = 2
                EVENTS[phase]['stim{:d}'.format(i+1)] = STIM_SECS
                EVENTS[phase]['postcue{:d}'.format(i+1)] = 1.5
                if proc[i+1] == 'CS+E':
                    EVENTS[phase]['iti{:d}'.format(i+1)] = itis[i] - 3.5
                else:
                    EVENTS[phase]['iti{:d}'.format(i+1)] = itis[i] - 1.5
    #add the initial iti to every run
    if DEV:
        EVENTS[phase]['iti0'] = 1
        EVENTS[phase].move_to_end('iti0',last=False)
    else:
        EVENTS[phase]['iti0'] = 10
        EVENTS[phase].move_to_end('iti0',last=False)

# screen
FRAME_RATE = 60. # flips per second
BKGRND_COLOR = 'black'

# keyboard
QUIT_KEY = '='
QUIT_MODIFIER = 'lshift'
SCAN_TRIGGER = '5'
BREAK_KEY = 'space'
RESP_KEYS = ['1', '2']


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
                'procedure']

    phases1_ = np.repeat([phase_names],np.tile(N_TRIALS,N_RUNS))
    events1_ = np.tile(np.array(range(1,N_TRIALS+1)),N_RUNS)
    tupes1 = list(zip(phases1_,events1_))

    phases2_ = np.repeat([phase_names],[73,73,97,97]) #these values have to be set manually bc of the pre and post cues during extinction
    events2_ = np.concatenate((list(EVENTS['fear1'].keys()),list(EVENTS['fear2'].keys()),list(EVENTS['ext1'].keys()),list(EVENTS['ext2'].keys())))
    tupes2 = list(zip(phases2_,events2_))


    df_indx = pd.MultiIndex.from_tuples(tupes1,names=['phase','trial'])
    df = pd.DataFrame(columns=df_cols,index=df_indx,dtype=float)
    df['subj'] = SUBJ
    df['sess'] = SESS
    df['date'] = DATE
    df['day1_block'] = 0
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
        df['day1_block'].loc[phase] = blocks[run]
        if 'fear' in phase:
            df['procedure'].loc[phase] = design['%s_shock'%(phase)].dropna().values
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


FIX_COLOR    = 'grey'
SUPP_COLOR   = 'blue'
#initialize the stimuli
fixStim   = visual.TextStim(win,'+',pos=[0,0],color=FIX_COLOR,height=2)
whiteBk   = visual.ImageStim(win,'whitebackground.jpg',pos=[0,0],size=[TARG_RAD*2,TARG_RAD*2])
fixBorder = visual.Rect(win,width=TARG_RAD*2,height=TARG_RAD*2,lineColor=FIX_COLOR,fillColor=None,lineWidth=8)
suppStim  = visual.TextStim(win,'+',pos=[0,0],color=SUPP_COLOR,height=2)
imgStim   = visual.ImageStim(win,pos=[0,0],size=[TARG_RAD*2,TARG_RAD*2])
suppBorder= visual.Rect(win,width=TARG_RAD*2,height=TARG_RAD*2,lineColor=SUPP_COLOR,fillColor=None,lineWidth=8)
txtStim   = visual.TextStim(win,pos=[0,0],height=.5) #,wrapWidth=req_dva_size)

#day1 response options
YES    = visual.TextStim(win,'100% \n Yes',pos=[-5,-8.15],height=.5)
NO     = visual.TextStim(win,'100% \n  No',pos=[5,-8.15],height=.5)
ZERO   = visual.TextStim(win,'0%',pos=[0,-7.8],height=.5)


expBk = visual.Rect(win,pos=[0,-6.5],width=TARG_RAD*2,height=2,lineColor='grey',fillColor=None,lineWidth=2)
expRect = visual.Rect(win,lineColor='grey',fillColor='grey',lineWidth=1)

#######################
##  event functions  ##
#######################

def instructions(phase):
    if phase in phase_names:
        msg = 'Respond on every trial whether or not you expect a shock\n(YES or NO).\n\nRemember, if the color of the cross (+) and border change to BLUE: try to clear the corresponding image from your mind.\n\nDo not think about other images, instead focus on mentally suppressing the current image.'
        txtStim.text = msg
        Event.clearEvents()
        txtStim.draw()
        win.flip()
        Event.waitKeys(keyList=['space'])


def show_event(event,phase,BIO):
    until = stamp_df.loc[(phase,event),'planned_offset']
    start = clock.getTime()

    SHOCK = False
    if 'stim' in event:
        #set image
        tnum = int(''.join([ x for x in event if x.isdigit() ]))
        img_fname = df.loc[(phase,tnum),'imgFname']
        imgStim.setImage(img_fname)
        
        # stim_list = [imgStim,fixBorder,YES,NO]
        stim_list = [imgStim,fixBorder,expBk,expRect,YES,NO,ZERO]
        #deliver the shock at the end of the trial if CSUS
        if 'fear' in phase and df.loc[(phase,tnum),'procedure'] == 'CSUS': SHOCK = True

        #change the border color if its CS+E during extinction
        if 'ext' in phase and 'E' in df.loc[(phase,tnum),'condition']: stim_list = [imgStim,suppBorder,expBk,expRect,YES,NO,ZERO]
            

    elif 'iti' in event:
        stim_list = [whiteBk,fixBorder,fixStim,expBk,YES,NO,ZERO]
    elif 'pre' in event:
        stim_list = [whiteBk,suppBorder,suppStim]
    elif 'post' in event:
        stim_list = [whiteBk,suppBorder,suppStim]
    else:
        stim_list = []

    stamp_onoffset('onset',stamp_idx=(phase,event),BIO=BIO,SHOCK=False)
    if 'stim' in event:
        t0, kp = prep_for_response()
        mouse.setPos()
        xPos = mouse.getPos()[0]

    while clock.getTime() < until:
        for stim in stim_list:
            stim.draw()
        win.flip(); check4quit()
        if 'stim' in event:
            if 1 not in kp:
                kp,rt = collect_response()

                move = mouse.getRel()
                # print(move)
                # if np.any(move):
                    # mouse.setPos((mouse.getPos()[0]+move[0]/10,0))
                xPos = mouse.getPos()[0]
                savevals = [xPos,rt]

                if xPos > ((TARG_RAD*2) / 2):
                    xPos = (TARG_RAD*2) / 2
                    if mouse.getPos()[0] < ((TARG_RAD*2) / 2) + .2:
                        mouse.setPos((xPos,0))

                if xPos < (-1 * (TARG_RAD*2) / 2):
                    xPos = -1 * (TARG_RAD*2) / 2
                    if mouse.getPos()[0] < (-1 * (TARG_RAD*2) / 2) - .2:
                        mouse.setPos((xPos,0))

                expRect.setVertices(((xPos,-7.5),(xPos,-5.5),(0,-5.5),(0,-7.5)))

    stamp_onoffset('offset',stamp_idx=(phase,event),BIO=BIO,SHOCK=SHOCK)

    if 'stim' in event:
        expRect.setVertices(((0,-7.5),(0,-5.5),(0,-5.5),(0,-7.5)))
        savecols = ['resp','rt']
        # if kp is rt:
        #     savevals = [0,0]
        print(savevals)
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

for phase in run_phases:
    # break message
    # if not DEV:
    instructions(phase)
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