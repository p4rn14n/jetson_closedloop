
import sys
sys.path.append('..')
from CameraFactory import CameraFactory
from imutils.video import FPS
import numpy as np
import cv2
import imutils
import time
from datetime import datetime
# from threading import *
import threading
#from gpiozero import LED
import Jetson.GPIO as GPIO
from Arduino import Arduino
import cvui
import os
import tables
import helper as clh
import roi_manager
from collections import deque
import re
import json
from configparser import ConfigParser
import csv
import board
import busio
import adafruit_mpr121
from audiostream import get_output
from audiostream.sources.wave import SineSource
import warnings
import random as rm
# import pyfirmata
import pandas as pd
from pymata4 import pymata4

warnings.filterwarnings("ignore", category=DeprecationWarning)


import threading
import time
import cv2


def mono12_to_uint8(img16):
    """
    Convert a Mono12 image stored in uint16 to uint8 for preview/video.
    Assumes useful range is 0..4095.
    """
    img8 = np.clip(img16, 0, 4095)
    img8 = (img8 * (255.0 / 4095.0)).astype(np.uint8)
    return img8

'''CHECKED'''
# --- NEW: simple block-wise spatial binning ---
def spatial_bin(img, bin_factor):
    """
    img: 2D array (H, W)
    returns: 2D array (H/bin_factor, W/bin_factor) with mean over each block.
    """
    h, w = img.shape
    h2 = (h // bin_factor) * bin_factor
    w2 = (w // bin_factor) * bin_factor
    img = img[:h2, :w2]
    img = img.reshape(h2 // bin_factor, bin_factor,
                      w2 // bin_factor, bin_factor).mean(axis=(1, 3))
    return img
# --- END NEW ---


'''CHECKED'''
configPath = '../config4.ini'
config      = ConfigParser()
config.read(configPath)
cfg = 'sentech_2roi'
cfgDict = dict(config.items(cfg))
vidSourceName = cfgDict['vid_source']
data_root   = cfgDict['data_root']
image_stream_filename = cfgDict['raw_image_file']
behavior_filename = cfgDict['video_file']
res     = list(map(int, cfgDict['resolution'].split(', ')))
res_b   = list(map(int, cfgDict['resolution_b'].split(', ')))
fr = int(cfgDict['framerate'])
fr_b = int(cfgDict['framerate_b'])
# --- NEW: binning settings for brain camera ---
BINNED_SIZE = 128          # final size for brain images
BIN_FACTOR = res[0] // BINNED_SIZE  # assuming square (512/128 = 4)
assert res[0] % BINNED_SIZE == 0 and res[1] % BINNED_SIZE == 0, "resolution must be divisible by BINNED_SIZE"
# --- END NEW ---

#dff_history is in seconds. Multiply with framerate to get array length
dffHistory = int(cfgDict['dff_history']) * fr
anchor = cvui.Point()
roi_operation_arr = re.split('([-+/%])',cfgDict['roi_operation'])
roi_names = re.split('[-+/%]',cfgDict['roi_operation'])
rois = []
ppmm = float(cfgDict['ppmm'])
# roi_size in config file is in mm. We convert it to pixel coordinates and round off.
roi_size = [int(round(x)) for x in np.array([float(i) for i in cfgDict['roi_size'].split(',')]) * ppmm]
# --- NEW: binned ROI size in pixels ---
roi_size_b = [max(1, s // BIN_FACTOR) for s in roi_size]
# --- END NEW ---

# total_trials = int(cfgDict['total_trials'])
# # total_trials = 25

maxTrialDur = int(cfgDict['max_trial_dur'])
rest_duration = int(cfgDict['initial_rest_dur'])
successrest_duration = int(cfgDict['success_rest_dur'])
failrest_duration = int(cfgDict['fail_rest_dur'])
bregma = list(map(int, cfgDict['bregma'].split(', ')))

# get seed pixel locations (in mm) from config file and parse to a dictionary
cfgDict['seeds_mm'] = json.loads(cfgDict['seeds_mm'])
seeds = clh.generate_seeds(clh.Position(bregma[0], bregma[1]), cfgDict['seeds_mm'], ppmm, 'u')

# --- NEW: convert bregma + seeds from full-res pixels -> binned pixels ---
def _down_coord(p):
    return int(round(p / BIN_FACTOR))

bregma_b = [_down_coord(bregma[0]), _down_coord(bregma[1])]
br = clh.Position(bregma_b[0], bregma_b[1])

seeds_binned = {}
for name, pos in seeds.items():
    seeds_binned[name] = {
        'ML': _down_coord(pos['ML']),
        'AP': _down_coord(pos['AP'])
    }
# --- END NEW ---

normal_seeds_x_R = []
normal_seeds_y_R = []
normal_seeds_x_L = []
normal_seeds_y_L = []

'''CHECKED'''
# --- NEW: running average over binned, mono brain images ---
runningAvg = np.zeros((BINNED_SIZE, BINNED_SIZE), dtype=np.float32)
runningImgQueue = deque(maxlen=10)
runningImgQueue.append(np.zeros_like(runningAvg))
runningImgQuROI = []
# --- END NEW ---


for name in roi_names:
    seed = seeds_binned[name]   # <-- use binned seeds
    rois.append(roi_manager.Rect(
        name,
        x=int(seed['ML'] - roi_size_b[0] / 2),
        y=int(seed['AP'] - roi_size_b[1] / 2),
        w=roi_size_b[0],
        h=roi_size_b[1],
        color=[0, 0, 255]
    ))
    runningImgQuROI.append(deque(maxlen=dffHistory))
    # store 2D float32 ROI images (binned)
    runningImgQuROI[-1].append(
        np.zeros((roi_size_b[1], roi_size_b[0]), dtype=np.float32)
    )

'''DIFFERENT, CHANGED'''
audio_tr_prob = float(config.get(cfg, 'audio'))
n_tones = int(config.get(cfg, 'n_tones'))
audiodelay = int(config.get(cfg, 'audio_delay'))
freqQue = deque(maxlen=(audiodelay*fr)+1)
freqQue.extend([600]*(audiodelay*fr))


# rewarddelay = int(config.get(cfg, 'reward_delay'))
# relAvgDffQue = deque(maxlen=(rewarddelay*fr)+1)
# relAvgDffQue.extend([0]*(rewarddelay*fr))

# rewarddelay_s = float(config.get(cfg, 'reward_delay'))  # seconds
# rewarddelay_s = float(0.2)
rewarddelay_s = float(config.get(cfg, 'reward_delay'))  # seconds
rewarddelay_frames = int(round(rewarddelay_s * fr))     # frames

relAvgDffQue = deque(maxlen=rewarddelay_frames + 1)
relAvgDffQue.extend([0] * rewarddelay_frames)


'''CHECKED'''
reward_threshold = float(config.get(cfg, 'reward_threshold'))
adaptive_threshold = int(config.get(cfg, 'adaptive_threshold'))
sessionType = clh.SessionType.normal_audio_normal_reward
# sessionType = clh.SessionType.no_audio_random_reward

task_mode = int(input("Baseline (0) or Training (1)?"))
# ---------------- BASELINE MODE ----------------
if task_mode == 0: #baseline recording
    BASELINE_MODE = True   # True = no audio, no reward, no punishment
    total_trials = 50 #50 trials 20 minutes
else:
    BASELINE_MODE = False
    total_trials = int(cfgDict['total_trials'])
    # total_trials = 25

BASELINE_LABEL = 'baseline_no_audio_no_reward'
BASELINE_SUCCESS_REST_DUR = failrest_duration   # keep rest structure simple after "virtual success"
# ----------------------------------------------

preview_every = 10

#list of sessions. session_name,reward_threshold
sessions = [('S1',reward_threshold)]

working = False

n_aud_ch = 1
au_rate = 44100
audio_tr_arr = rm.sample([1]*int(audio_tr_prob*total_trials) + [0]*int((1-audio_tr_prob)*total_trials),k=total_trials)

#get frequencies in range 1-24k with quarter octave increment
freqs = [1000 * (2**(1/4))**i for i in range(n_tones)]


dff_bins = np.linspace(0, reward_threshold, len(freqs)-2)
## set lower and upper limits too high so dff indexing never goes out of range
dff_bins = np.insert(dff_bins, 0, np.NINF)
dff_bins = np.append(dff_bins, np.inf)

ts_trigger = 0
ts_detection = 0
run_threads = True
runPreview = True
runRecording = True


ledReward = 12
ledLightTTL = 13


board = pymata4.Pymata4()


board.set_pin_mode_digital_output(ledReward)
board.set_pin_mode_digital_output(ledLightTTL)




mouse_id = input("Please enter mouse ID: ")

WINDOW_NAME = 'Closed loop'
DFF_WINDOW = 'Corrected DFF'
BEH_WINDOW = 'Behavior'

cvui.init(WINDOW_NAME)
cvui.init(DFF_WINDOW)
cvui.init(BEH_WINDOW)

'''DIFFERENT'''

roi_manager.init(roi_manager.annots, rois, WINDOW_NAME, BINNED_SIZE, BINNED_SIZE)

cv2.setMouseCallback(WINDOW_NAME, roi_manager.dragcircle, roi_manager.annots)


    
def blink_led_pymata(led, t):
    board.digital_write(led,1)
    time.sleep(t)
    board.digital_write(led, 0)


'''CHECKED'''
summary_filename = data_root + os.sep + config.get(cfg, 'summary_file')
summary_exists = os.path.isfile(summary_filename)
summaryfile = open (summary_filename, 'a', encoding="utf-8")
headers = [col.strip() for col in config.get(cfg, 'summary_header').split(',')]
writer = csv.DictWriter(summaryfile, delimiter=',', lineterminator='\n',fieldnames=headers)
if not summary_exists:
    writer.writeheader()  # file doesn't exist yet, write a header

print("Start preview\n\n")
# initialize the camera and grab a reference to the raw camera capture
exec('from ' + vidSourceName + ' import ' + vidSourceName)
# initialize the camera and grab a reference to the raw camera capture


# initialize the camera and grab a reference to the raw camera capture
deviceIdx = 1
vs1 = CameraFactory(eval(vidSourceName+'(cfgDict, deviceIdx).start()'))
image1 = vs1.get_image()
deviceIdx = 0
vs2 = CameraFactory(eval(vidSourceName+'(cfgDict, deviceIdx).start()'))
image2 = vs2.get_image()


# allow the camera to warmup
time.sleep(2)


board.digital_write(ledLightTTL,1)
last_time=time.time()

'''DIFFERENT'''
# t_last = None


'''CHECKED'''
while runPreview:
    start = time.time()
    

    # Behavior camera (Mono8, 2D)
    image1 = vs1.get_image()
    if image1 is None:
        print('Image1 none')
        continue
    image1 = image1[-res_b[1]:, -res_b[0]:]   # crop to desired ROI (2D uint8)

    # Behavior image: Mono8 -> BGR
    image1_8 = mono12_to_uint8(image1)
    beh_display = cv2.cvtColor(image1_8, cv2.COLOR_GRAY2BGR)

    img2 = vs2.get_image()
    if img2 is None:
        print('Image2 none')
        continue
    # last_fid2 = fid2
    img2 = img2[-res[1]:, -res[0]:]           # crop to 512×512

    # --- NEW: spatially bin brain image to 128×128 and cast to float32 ---
    image2 = spatial_bin(img2, BIN_FACTOR).astype(np.float32)
    image2_clean = image2.copy()
    # --- END NEW ---

    
    runningImgQueue.append(image2)
    runningAvg = np.mean(runningImgQueue, axis=0)

    # ΔF/F on binned image (2D)
    dff = image2 - runningAvg
    dffCorrected = np.divide(
        dff, runningAvg,
        out=np.zeros_like(dff),
        where=runningAvg != 0
    )

    '''CHANGED A LOT'''

    # Render the roi
    # for roi in rois:
    #     cv2.rectangle(dffCorrected, (roi.x, roi.y), (roi.x+roi.w, roi.y+roi.h), roi.color, 1)
    #     cropRoi = dffCorrected[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    #     roi.avgDff = np.sum(cropRoi)/roi.area()

    # compute roi.avgDff from the clean DFF
    for roi in rois:
        cropRoi = dffCorrected[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]
        roi.avgDff = np.sum(cropRoi)/roi.area()


    relAvgDff = rois[0].avgDff

    
        # --- LUMINANCE LINE FOR LED TUNING (use clean image) ---


    mid_row_c = image2_clean.shape[0] // 2
    lum_line_c = image2_clean[mid_row_c, :].copy()

    def roi_max_lum(img, roi):
        y0 = max(0, roi.y)
        x0 = max(0, roi.x)
        y1 = min(img.shape[0], roi.y + roi.h)
        x1 = min(img.shape[1], roi.x + roi.w)
        if y1 <= y0 or x1 <= x0:
            return float('nan')
        return float(np.max(img[y0:y1, x0:x1]))
 
    roi1_max_c = roi_max_lum(image2_clean, rois[0])
    roi2_max_c = roi_max_lum(image2_clean, rois[1])



    brain_norm = image2_clean - np.min(image2_clean)
    brain_norm = brain_norm / (np.max(brain_norm) + 1e-6)
    brain_display = (brain_norm * 255).astype('uint8')
    brain_display = cv2.cvtColor(brain_display, cv2.COLOR_GRAY2BGR)

    # This function must be called *AFTER* all UI components. It does
    # all the behind the scenes magic to handle mouse clicks, etc.



    # --- draw bregma + seeds on brain_display ONLY ---
    brain_display[br.row, br.col] = (0, 0, 255)
    for seedname in seeds_binned:
        ap = seeds_binned[seedname]['AP']
        ml = seeds_binned[seedname]['ML']
        brain_display[ap, ml] = (0, 255, 0)
        if "_R" in seedname:
            normal_seeds_x_R.append(ml)
            normal_seeds_y_R.append(ap)
        else:
            normal_seeds_x_L.append(ml)
            normal_seeds_y_L.append(ap)
    # --- end markers ---


    '''NEWNEWNEWNEW '''
    for roi in rois:
            # draw rectangles on a uint8 display image (NOT on dffCorrected)
        cv2.rectangle(brain_display, (roi.x, roi.y),
                    ((roi.x+roi.w), (roi.y+roi.h)), (0,0,255), 1)
        

    FONT = 0.3  # try 0.35–0.55

    cvui.update()

    cvui.printf(brain_display, 5, 5, FONT, 0xFF0000, "(midline): %.1f" % lum_line_c.max())
    cvui.printf(brain_display, 5, 108, FONT, 0xFF0000, "(%s): %.1f" % (rois[0].name, roi1_max_c))
    cvui.printf(brain_display, 5, 118, FONT, 0xFF0000, "(%s): %.1f" % (rois[1].name, roi2_max_c))

    # --- END NEW ---

    dff_disp = np.clip(dffCorrected, -0.15, 0.15)
    dff_disp = dff_disp - np.min(dff_disp)
    dff_disp = dff_disp / (np.max(dff_disp) + 1e-6)
    dff_display = (dff_disp * 255).astype('uint8')
    
    dff_display = cv2.cvtColor(dff_display, cv2.COLOR_GRAY2BGR)



    scale = 4  # enlarge 128x128 -> 512x512 on screen
    brain_display = cv2.resize(brain_display, (BINNED_SIZE*scale, BINNED_SIZE*scale), interpolation=cv2.INTER_NEAREST)
    dff_display   = cv2.resize(dff_display,   (BINNED_SIZE*scale, BINNED_SIZE*scale), interpolation=cv2.INTER_NEAREST)

    cvui.update()
    cvui.imshow(BEH_WINDOW, beh_display)
    cvui.imshow(WINDOW_NAME, brain_display)
    cvui.imshow(DFF_WINDOW, dff_display)


    '''DIFFERENT BUT I DONT KNOW WHAT TO DO ABOUT IT'''
    # Press Esc or Ctrl-C to stop the program
    k = cv2.waitKey(1)
    if k == 27:
        runPreview = False
        break

    time.sleep(max(1./fr - (time.time() - start), 0))
    print('FPS', round(1 / (time.time() - last_time), 2), end='\r')
    last_time=time.time() 



for roi in rois:
    config.set(cfg, roi.name,str(roi.x)+','+str(roi.y))
#co save the configuration used in current directory
with open(configPath, 'w', encoding="utf-8") as f:
        config.write(f)

cv2.destroyWindow(DFF_WINDOW)


board.digital_write(ledLightTTL,0)

time.sleep(2)

# Get the current time and initialize the project folder
tm = datetime.now()
data_name = str(tm.year) + \
            format(tm.month, '02d') + \
            format(tm.day, '02d') + \
            format(tm.hour, '02d') + \
            format(tm.minute, '02d') + \
            format(tm.second, '02d')
data_root = data_root + os.sep + mouse_id + os.sep + data_name
if not os.path.exists(data_root):
        print("Creating data directory: ",data_root)
        os.makedirs(data_root)

for session, reward_threshold in sessions:
    #breakpoint()
    
    if runRecording:

        totRewards = 0

        # Get the current time and initialize the project folder
        tm = datetime.now()
        session_root = data_root + os.sep + session
        if not os.path.exists(session_root):
                print("Creating data directory: ",session_root)
                os.makedirs(session_root)

        # allow the camera to warmup
        time.sleep(2)
        
        image1 = vs1.get_image()
        image1 = image1[-res_b[1]:, -res_b[0]:]      # 2D, uint8

        time.sleep(1) # Let behavior recording start first
        
        # Get one frame from each camera to define shapes
        raw2 = vs2.get_image()
        raw2 = raw2[-res[1]:, -res[0]:]              # 512x512, uint16
        raw2_binned = spatial_bin(raw2, BIN_FACTOR).astype(np.uint16)  # 128x128, uint16
        rows, cols = raw2_binned.shape


        
        config.set(cfg, 'reward_threshold',str(reward_threshold))
        config.set('configsection', 'config', cfg)
        # save the configuration used in target data directory
        with open(session_root + '/config.ini', 'w', encoding="utf-8") as f:
                config.write(f)
        image_hdf5_path = session_root + os.sep + mouse_id + '_' + data_name + image_stream_filename
        image_hdf5_file = tables.open_file(image_hdf5_path, mode='w')
        # --- NEW: store binned mono brain frames, 128x128, uint16 ---
        image_storage = image_hdf5_file.create_earray(
            image_hdf5_file.root,
            'raw_images',
            tables.UInt16Atom(),
            shape=(0, BINNED_SIZE, BINNED_SIZE)
        )

        video_path = session_root + os.sep + mouse_id + '_' + data_name + behavior_filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer1= cv2.VideoWriter(video_path, fourcc, fr_b, [res_b[0], res_b[1]])
        ''''''

        logFileName = session_root + os.sep + mouse_id + '_' + data_name + '.txt'
        logFile = open(logFileName, 'w', encoding="utf-8")
        logFile.write('frame' + '\t' +
                      'time' + '\t' +
                      roi_operation_arr[0] + 'dff' + '\t' +
                      roi_operation_arr[2] + 'dff' + '\t' +
                      roi_operation_arr[0] + 'dff' +
                      roi_operation_arr[1] +
                      roi_operation_arr[2] + 'dff' + '\t' +
                      'freq' + '\t' +
                      'rew_threshold' + '\t' +
                      'reward' + '\t' +
                      'trial' + '\t' +
                      'audio' + '\t' +
                      'tot_rewards' + '\t' +
                      'baseline_mode' + '\n')
                    #   roi_operation_arr[0] + 'F0' + '\t' + 
                    #   roi_operation_arr[2] + 'F0' + '\n') 
                    #   roi_operation_arr[0] + 'F1' + '\t' + 
                    #   roi_operation_arr[2] + 'F1' + '\t' + 
                    #   roi_operation_arr[0] + 'noiseSTD' + '\t' + 
                    #   roi_operation_arr[2] + 'noiseSTD' + '\n')
                    #   'lick' + '\n')
        print("Start recording\n")
        
    
    image2 = vs2.get_image()
    image2 = image2[-res[1]:, -res[0]:]
    
    # time.sleep(2)

    board.digital_write(ledLightTTL,1)

    fps = FPS().start()

    restTimer = time.time()
    rest = True
    runTrial = False
    nTrial = 0
    rewardsInEpoch = 0
    rewardThreshTimer = time.time()
    lastRewTime = time.time()
    
    ####################
    # get a output stream where we can play samples
    austream = get_output(channels=n_aud_ch, rate=au_rate, buffersize=1024)
    # create one wave sin() at 220Hz, attach it to our speaker, and play
    sinsource = SineSource(austream, 1000)
    sinsource.stop()
    #############
    
    '''DIFFERENT'''
    # t_last = None

    trial_peak_rel = -np.inf
    trial_start_time = None
    baseline_trial_summary = []
    
    '''CHECKED, IMAGE1 NOT STARTED'''
    while image2 is not None and nTrial <= total_trials:
        start = time.time()
        
        '''MOVED BEHAVIOR HERE'''
        # Behavior camera (Mono8)
        image1 = vs1.get_image()
        image1 = image1[-res_b[1]:, -res_b[0]:]   # 2D uint8


        # Brain camera (Mono12)
        raw2 = vs2.get_image()

        '''WAS CONTINUE HERE. CHANGED IT TO BREAK'''
        if raw2 is None:
            break
        # last_fid2 = fid2
        raw2 = raw2[-res[1]:, -res[0]:]           # 512x512 uint16


        # --- NEW: bin and cast for dF/F ---
        raw2_binned = spatial_bin(raw2, BIN_FACTOR).astype(np.uint16)   # 128x128, uint16
        image2 = raw2_binned.astype(np.float32)                         # dF/F will use this
        # --- END NEW ---

        reward = 0
        touch = 0

        fps.update()


        '''TIME THINGY IS NEW HERE'''
        # ---- Brain camera FPS diagnostic ----
        # now = time.time()

        # if t_last is None:
        #     t_last = now
        # else:
        #     dt = now - t_last
        #     if dt >= 1.0:
        #         t_last = now

        # ------------------------------------

        '''CHECKED'''
        for roi, imQ in zip(rois, runningImgQuROI):
            imROI = image2[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]   # 2D float32

            runAvg = np.mean(imQ, axis=0)
            dffROI = imROI - runAvg
            dffROI = np.divide(
                dffROI, runAvg,
                out=np.zeros_like(dffROI),
                where=runAvg != 0
            )
            dffCorrected = dffROI
            imQ.append(imROI)
            roi.avgDff = np.sum(dffCorrected) / roi.area()

        
        relAvgDff = eval('rois[0].avgDff'+roi_operation_arr[1]+'rois[1].avgDff')
        relAvgDffQue.append(relAvgDff)
        relAvgDffDelayed = relAvgDffQue.popleft()
        
        fq = int(freqs[np.searchsorted(dff_bins, relAvgDff)])
        freqQue.append(fq)
        freq = freqQue.popleft()

        ''''''

        if runTrial:
            if trial_start_time is None:
                trial_start_time = time.time()
                trial_peak_rel = -np.inf

            if relAvgDffDelayed > trial_peak_rel:
                trial_peak_rel = relAvgDffDelayed

            tr = nTrial
            rest_duration = failrest_duration

            # default: no real reward delivered
            deliverReward = False

            if not BASELINE_MODE:
                if sessionType == clh.SessionType.normal_audio_normal_reward:
                    deliverReward = relAvgDffDelayed > reward_threshold
                elif sessionType == clh.SessionType.no_audio_random_reward:
                    deliverReward = np.random.choice(np.arange(0, 2), p=[0.998, 0.002])

            if deliverReward:
                reward = 1
                totRewards += 1
                rewardsInEpoch += 1

                print("RelAvgDff is: ", relAvgDffDelayed)
                print("Rew threshold is: ", reward_threshold)
                print('total rewards: ' + str(totRewards), end="\n")

                t2 = threading.Thread(target=blink_led_pymata, args=(ledReward, 0.12))
                t2.start()

                lastRewTime = time.time()
                runTrial = False
                rest_duration = successrest_duration


                baseline_trial_summary.append({
                    'trial': tr,
                    'peak_relAvgDff': trial_peak_rel,
                    'duration_s': time.time() - trial_start_time
                })
                trial_peak_rel = -np.inf
                trial_start_time = None

            elif time.time() - trialTimer >= maxTrialDur:
                runTrial = False

                # In baseline mode, trial just times out naturally; don't mark punishment.
                if BASELINE_MODE:
                    reward = 0
                    baseline_trial_summary.append({
                        'trial': tr,
                        'peak_relAvgDff': trial_peak_rel,
                        'duration_s': time.time() - trial_start_time
                    })
                    trial_peak_rel = -np.inf
                    trial_start_time = None
                else:
                    reward = -1

            restTimer = time.time()

            if not BASELINE_MODE:
                sinsource.frequency = freq

        else:
            if not rest:
                if not BASELINE_MODE:
                    sinsource.stop()
                print('Trial end')

            rest = True
            runTrial = False

            if rest and time.time() - restTimer < rest_duration:
                freq = 0
                tr = 0
                audio = 0
                trialTimer = time.time()

            else:
                rest = False
                runTrial = True

                if BASELINE_MODE:
                    audio = 0
                else:
                    audio = audio_tr_arr[tr]

                if (not BASELINE_MODE) and audio:
                    austream = get_output(channels=n_aud_ch, rate=au_rate, buffersize=1024)
                    sinsource = SineSource(austream, freq)
                    sinsource.start()

        if adaptive_threshold and time.time() - rewardThreshTimer > 30:
            # if more than 1 reward dispensed in last epoch, increase the threshold
            if rewardsInEpoch > 1:
                reward_threshold += 0.002
            # if no reward was dispensed in last epoch, decrease the threshold
            if rewardsInEpoch ==0:
                reward_threshold -= 0.002

            rewardsInEpoch = 0
            rewardThreshTimer = time.time()
            # regenerate the dff bins to frequency mappings
            dff_bins = np.linspace(-reward_threshold, reward_threshold, len(freqs)-2)
            dff_bins = np.insert(dff_bins, 0, np.NINF)
            dff_bins = np.append(dff_bins, np.inf)
        
        if tr ==0 and runTrial:
            nTrial += 1
            print("Trial: ",nTrial)
        
        # Save binned raw brain image (uint16) and behavior video frame
        image_storage.append(raw2_binned[None])   # 1 x 128 x 128
        # Behavior video needs 8-bit 3-channel

        '''DIFFEREBT. HERE WE DON'T WRITE VIDEOWRITER1'''
        # video_writer1.write(cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR))
        image1_8 = mono12_to_uint8(image1)
        video_writer1.write(cv2.cvtColor(image1_8, cv2.COLOR_GRAY2BGR))
        
        sttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logFile.write(str(fps._numFrames) + '\t' +
                      sttime + '\t' +
                      str(rois[0].avgDff) + '\t' +
                      str(rois[1].avgDff) + '\t' +
                      str(relAvgDff) + '\t' +
                      str(freq) + '\t' +
                      str(reward_threshold) + '\t' +
                      str(reward) + '\t' +
                      str(tr) + '\t' +
                      str(audio) + '\t' +
                      str(totRewards) + '\t' +
                      str(int(BASELINE_MODE)) + '\n' )

        
        # beh_display = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)


        # # Display brain (normalized, binned)
        # brain_norm = image2 - np.min(image2)
        # brain_norm = brain_norm / (np.max(brain_norm) + 1e-6)
        # brain_display = (brain_norm * 255).astype('uint8')
        # brain_display = cv2.cvtColor(brain_display, cv2.COLOR_GRAY2BGR)

        # scale = 4
        # brain_display = cv2.resize(brain_display, (BINNED_SIZE*scale, BINNED_SIZE*scale), interpolation=cv2.INTER_NEAREST)

        # '''BEH-DISPLAY HERE IS A COUPLE OF LINES ABOVE'''
        # cvui.imshow(BEH_WINDOW, beh_display)

        # ''''''
        # cvui.imshow(WINDOW_NAME, brain_display)


        frame_count = fps._numFrames
        if frame_count % preview_every == 0:
            image1_8 = mono12_to_uint8(image1)
            beh_preview = cv2.resize(image1_8, (160, 120), interpolation=cv2.INTER_AREA)
            beh_preview = cv2.cvtColor(beh_preview, cv2.COLOR_GRAY2BGR)
            cvui.imshow(BEH_WINDOW, beh_preview)

            # for brain, maybe don't normalize every frame
            brain_preview = cv2.resize(image2, (128, 128), interpolation=cv2.INTER_NEAREST)
            brain_preview = cv2.normalize(brain_preview, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            brain_preview = cv2.cvtColor(brain_preview, cv2.COLOR_GRAY2BGR)
            cvui.imshow(WINDOW_NAME, brain_preview)


        # Press Esc or Ctrl-C to stop the program
        k = cv2.waitKey(1)
        if k == 27:
            break
        
        '''DIFFERENT HERE: COMMENTED THIS LINE'''
        time.sleep(max(1./fr - (time.time() - start), 0))

        
    '''CHECKED'''
    board.digital_write(ledLightTTL,0)



    """record for some time after"""


    t_end = time.time() + 4
    while time.time() < t_end:
        start = time.time()
        image1 = vs1.get_image()
        image1 = image1[-res_b[1]:, -res_b[0]:]   # 2D uint8

        # Brain camera (Mono12)
        raw2 = vs2.get_image()
        raw2 = raw2[-res[1]:, -res[0]:]           # 512x512 uint16

        # --- NEW: bin and cast for dF/F ---
        raw2_binned = spatial_bin(raw2, BIN_FACTOR).astype(np.uint16)   # 128x128, uint16
        # image2 = raw2_binned.astype(np.float32)    

        fps.update()
        
        # video_writer1.write(image1)
        image1_8 = mono12_to_uint8(image1)
        video_writer1.write(cv2.cvtColor(image1_8, cv2.COLOR_GRAY2BGR))
        image_storage.append(raw2_binned[None])
        sttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logFile.write(str(fps._numFrames) + '\t' +
                      sttime + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\n')
    

        # beh_display = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        # brain_norm = image2 - np.min(image2)
        # brain_norm = brain_norm / (np.max(brain_norm) + 1e-6)
        # brain_display = (brain_norm * 255).astype('uint8')
        # brain_display = cv2.cvtColor(brain_display, cv2.COLOR_GRAY2BGR)
        # scale = 4
        # brain_display = cv2.resize(brain_display, (BINNED_SIZE*scale, BINNED_SIZE*scale), interpolation=cv2.INTER_NEAREST)

        # cvui.imshow(BEH_WINDOW, beh_display)
        # cvui.imshow(WINDOW_NAME, brain_display)
        # frame_count = fps._numFrames
        # if frame_count % preview_every == 0:
        #     beh_preview = cv2.resize(image1, (160, 120), interpolation=cv2.INTER_AREA)
        #     cvui.imshow(BEH_WINDOW, beh_preview)

        #     # for brain, maybe don't normalize every frame
        #     brain_preview = cv2.resize(image2, (128, 128), interpolation=cv2.INTER_NEAREST)
        #     brain_preview = cv2.normalize(brain_preview, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #     cvui.imshow(WINDOW_NAME, brain_preview)


        k = cv2.waitKey(1)
        if k == 27:
            break
        time.sleep(max(1./fr - (time.time() - start), 0))


    
    fps.stop()

    '''HERE VS1.STOP AND VS2.STOP COME AFTER THAN RELEASE..'''

    '''CHECKED'''
    print('total rewards: ' + str(totRewards))
    
    # time.sleep(3) # let vs stop and behavior recording capture light off

    
    image_hdf5_file.close()
    
    # video_writer1.release()

    '''SOME NEW STUFF HERE'''

    video_writer1.release()

    vs1.stop()
    vs2.stop()


    '''CHECKED'''
    logFile.close()

    writer.writerow({headers[0]: mouse_id,
                     headers[1]: session,
                     headers[2]: session_root,
                     headers[3]: fps._start,
                     headers[4]: fps._end,
                     headers[5]: fps.elapsed(),
                     headers[6]: fps.fps(),
                     headers[7]: dffHistory,
                     headers[8]: n_tones,
                     headers[9]: reward_threshold,
                     headers[10]: totRewards,
                     headers[11]: audio_tr_prob,
                     headers[12]: sessionType.name})
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}\n\n".format(fps.fps()))

run_threads = False

summaryfile.close()

sinsource.stop()

cv2.destroyAllWindows()

if len(baseline_trial_summary) > 0:
    baseline_csv = session_root + os.sep + mouse_id + '_' + data_name + '_baseline_trial_summary.csv'
    pd.DataFrame(baseline_trial_summary).to_csv(baseline_csv, index=False)