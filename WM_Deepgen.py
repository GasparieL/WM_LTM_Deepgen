#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Wed Aug  7 17:57:56 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'WM_Deepgen'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [3200, 1800]
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/lgaspari/wm_deepgen/WM_Deepgen.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
        )
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
        )
    if deviceManager.getDevice('key_resp_16') is None:
        # initialise key_resp_16
        key_resp_16 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_16',
        )
    if deviceManager.getDevice('key_resp_13') is None:
        # initialise key_resp_13
        key_resp_13 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_13',
        )
    if deviceManager.getDevice('key_resp_14') is None:
        # initialise key_resp_14
        key_resp_14 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_14',
        )
    if deviceManager.getDevice('key_resp_15') is None:
        # initialise key_resp_15
        key_resp_15 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_15',
        )
    if deviceManager.getDevice('key_resp_6') is None:
        # initialise key_resp_6
        key_resp_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_6',
        )
    if deviceManager.getDevice('key_resp_7') is None:
        # initialise key_resp_7
        key_resp_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_7',
        )
    if deviceManager.getDevice('key_resp_8') is None:
        # initialise key_resp_8
        key_resp_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_8',
        )
    if deviceManager.getDevice('key_resp_9') is None:
        # initialise key_resp_9
        key_resp_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_9',
        )
    if deviceManager.getDevice('key_resp_11') is None:
        # initialise key_resp_11
        key_resp_11 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_11',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_12') is None:
        # initialise key_resp_12
        key_resp_12 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_12',
        )
    if deviceManager.getDevice('key_resp_17') is None:
        # initialise key_resp_17
        key_resp_17 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_17',
        )
    if deviceManager.getDevice('image_recall_response') is None:
        # initialise image_recall_response
        image_recall_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='image_recall_response',
        )
    if deviceManager.getDevice('key_resp_18') is None:
        # initialise key_resp_18
        key_resp_18 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_18',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "setup_js" ---
    # Run 'Begin Experiment' code from code
    
    // inserted the crib sheet stuff
    thisExp=psychoJS.experiment;
    win=psychoJS.window;
    event=psychoJS.eventManager;
    
    
    //hide mouse
    document.body.style.cursor='none'
    
    //exit if on mobile
    if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
      quitPsychoJS('Mobile device detected. Goodbye!', false)
    }
    
    # --- Initialize components for Routine "Instructions1" ---
    text1 = visual.TextStim(win=win, name='text1',
        text='Welcome to the experiment! Please read the following instructions carefully. You will see a demo of the task first, so do not worry if you do not understand the task right away.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    text = visual.TextStim(win=win, name='text',
        text='[Press the "y" key to continue]',
        font='Open Sans',
        pos=(0, -.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Instructions2" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text="Please perform the experiment in full-screen mode until the experiment concludes. If you would like to discontinue participation, you may press the 'escape' key and the experiment will terminate. \n\nYou can avoid uninentionally ending the experiment by remaining in full-screen mode, even during breaks.",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    text_3 = visual.TextStim(win=win, name='text_3',
        text='[Press the "y" key to continue]',
        font='Open Sans',
        pos=(0, -.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Instructions3" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='When the task begins, please remain focused on the cross in the middle of the screen. Use your peripheral vision to attend to the items that appear on screen.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    text_5 = visual.TextStim(win=win, name='text_5',
        text='[Press the "y" key to continue]',
        font='Open Sans',
        pos=(0, -.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Instructions4" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='In this task, you will be remembering images.\n\nOn each trial, will see TWO images, one on the left and one on the right. They will flash on the screen quickly. Do your best to pay equal attention to both images. \n\nYou should try to keep the two images in mind even after they disappear. \n\nYou will be tested on either the left or the right image, but never on both.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    text_7 = visual.TextStim(win=win, name='text_7',
        text='[Press the "y" key to continue]',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Instructions5" ---
    text_16 = visual.TextStim(win=win, name='text_16',
        text='These cues tell you which image you are being tested on. For example, this cue indicates that you should select which image was originally on the left:\n',
        font='Open Sans',
        pos=(0, .3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='stimuli/cue_left.png', mask=None, anchor='center',
        ori=0.0, pos=(0, .15), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text_17 = visual.TextStim(win=win, name='text_17',
        text='This cue indicates that you should select the stimulus that was on the right:',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='stimuli/cue_right.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -.15), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    text_18 = visual.TextStim(win=win, name='text_18',
        text='[Press "y" to continue". ',
        font='Open Sans',
        pos=(0,-.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    
    # --- Initialize components for Routine "Instructions6b" ---
    text_25 = visual.TextStim(win=win, name='text_25',
        text='Get ready to try a few example trials.\n\nRemember, the arrows point to where the image you are being tested on was originally presented. Click on that image regardless of where on the screen it is during the test.\n\nIt takes a little practice to get used to the task rules. You may neeed to perform the following practice several times. This is perfectly normal. The practice will automatically restart until you respond correctly to enough of the trials.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_16 = keyboard.Keyboard(deviceName='key_resp_16')
    text_26 = visual.TextStim(win=win, name='text_26',
        text='[Press "y" to continue". ',
        font='Open Sans',
        pos=(0,-.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Instructions6" ---
    text_20 = visual.TextStim(win=win, name='text_20',
        text='After you see the two images, you will see a screen with 6 images to select between. Click on the image that was presented on the side of the screen that the arrow is pointing to. For example, if the arrow points left, click the image that was presented on the left side. \n\nAfter you make your response, a green box will highlight the correct answer.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_13 = keyboard.Keyboard(deviceName='key_resp_13')
    text_21 = visual.TextStim(win=win, name='text_21',
        text='[Press the "y" key to continue]',
        font='Open Sans',
        pos=(0, -.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "memoranda" ---
    left_stim = visual.ImageStim(
        win=win,
        name='left_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-.35, 0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_stim = visual.ImageStim(
        win=win,
        name='right_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(.35,0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    fixation_cross_5 = visual.TextStim(win=win, name='fixation_cross_5',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "isi1" ---
    fixation_cross = visual.TextStim(win=win, name='fixation_cross',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "probe" ---
    picker = visual.Rect(
        win=win, name='picker',
        width=(0.26, 0.26)[0], height=(0.26, 0.26)[1],
        ori=0.0, pos=(2, 2), anchor='center',
        lineWidth=10.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    distractor_mem_stim = visual.ImageStim(
        win=win,
        name='distractor_mem_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    memoranda_stim = visual.ImageStim(
        win=win,
        name='memoranda_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    distractor_im_0 = visual.ImageStim(
        win=win,
        name='distractor_im_0', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    distractor_im_1 = visual.ImageStim(
        win=win,
        name='distractor_im_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    distractor_im_2 = visual.ImageStim(
        win=win,
        name='distractor_im_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    distractor_im_3 = visual.ImageStim(
        win=win,
        name='distractor_im_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    mem_cue = visual.ImageStim(
        win=win,
        name='mem_cue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    text_24 = visual.TextStim(win=win, name='text_24',
        text='',
        font='Open Sans',
        pos=(0, .09), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "correct_ans" ---
    correct_border = visual.Rect(
        win=win, name='correct_border',
        width=(0.26, 0.26)[0], height=(0.26, 0.26)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=10.0,     colorSpace='rgb',  lineColor=[-0.5294, 0.4039, -0.1137], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    distractor_mem_stim_3 = visual.ImageStim(
        win=win,
        name='distractor_mem_stim_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    memoranda_stim_2 = visual.ImageStim(
        win=win,
        name='memoranda_stim_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    distractor_im = visual.ImageStim(
        win=win,
        name='distractor_im', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    distractor_im_4 = visual.ImageStim(
        win=win,
        name='distractor_im_4', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    distractor_im_5 = visual.ImageStim(
        win=win,
        name='distractor_im_5', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    distractor_im_6 = visual.ImageStim(
        win=win,
        name='distractor_im_6', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    distractor_im_7 = visual.ImageStim(
        win=win,
        name='distractor_im_7', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    mem_cue_2 = visual.ImageStim(
        win=win,
        name='mem_cue_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    
    # --- Initialize components for Routine "ITI" ---
    fixation_iti = visual.TextStim(win=win, name='fixation_iti',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "repeat_code" ---
    repeat_prac_text_2 = visual.TextStim(win=win, name='repeat_prac_text_2',
        text="You made too many incorrect responses during the practice.\n\nRemember, the arrows indicate the side where the image you are being tested on was originally presented. Click on that image regardless of where on the screen it is during the test.\n\nPlease try the practice again before moving on to the main experiment.\n\nPress 'y' to continue.",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_14 = keyboard.Keyboard(deviceName='key_resp_14')
    
    # --- Initialize components for Routine "Instructions7" ---
    text_22 = visual.TextStim(win=win, name='text_22',
        text='After you see the two images, a cue will appear that will tell you whether it is more likely that the stimulus on the left or the right will be tested.\n\nYou can use this information to help you perform the task well. ',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_15 = keyboard.Keyboard(deviceName='key_resp_15')
    text_23 = visual.TextStim(win=win, name='text_23',
        text="[Press 'y' to continue]",
        font='Open Sans',
        pos=(0, -.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Instructions8" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text='Take a look at this example. Most of the left side of the rectangle is filled in, indicating a higher probability that the left image will be tested. \n\n\n\nA small portion of the right side of the rectangle is filled in. This means there is a small possibility that the right image will be tested. \n\nThis information can help you perform the task.',
        font='Open Sans',
        pos=(0, .065), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_6 = keyboard.Keyboard(deviceName='key_resp_6')
    fill_rectangle2 = visual.Rect(
        win=win, name='fill_rectangle2',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, .1), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    right_cue2 = visual.Rect(
        win=win, name='right_cue2',
        width=(.01375, 0.049)[0], height=(.01375, 0.049)[1],
        ori=0.0, pos=(0.006875, .1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    left_cue_2 = visual.Rect(
        win=win, name='left_cue_2',
        width=(.15125, 0.049)[0], height=(.15125, 0.049)[1],
        ori=0.0, pos=(-0.075625, .1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    dividing_line_2 = visual.Line(
        win=win, name='dividing_line_2',
        start=(-(0.1, 0.1)[0]/2.0, 0), end=(+(0.1, 0.1)[0]/2.0, 0),
        ori=90.0, pos=(0, .1), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor=[0.3255, 0.3255, 0.3255], fillColor=[0.3255, 0.3255, 0.3255],
        opacity=None, depth=-5.0, interpolate=True)
    key_resp_7 = keyboard.Keyboard(deviceName='key_resp_7')
    text_9 = visual.TextStim(win=win, name='text_9',
        text="[Press 'y' to continue]",
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    
    # --- Initialize components for Routine "Instructions9" ---
    text_10 = visual.TextStim(win=win, name='text_10',
        text='You will see four different symbols:\n\nThis one, which you just saw, indicates that the left image is MUCH more likely than the right image to be tested. ',
        font='Open Sans',
        pos=(0, .3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    fill_rectangle_5 = visual.Rect(
        win=win, name='fill_rectangle_5',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, .1), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    right_cue3 = visual.Rect(
        win=win, name='right_cue3',
        width=(0.01375, 0.049)[0], height=(0.01375, 0.049)[1],
        ori=0.0, pos=(.006875, .1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    left_cue3 = visual.Rect(
        win=win, name='left_cue3',
        width=(0.15125, 0.049)[0], height=(0.15125, 0.049)[1],
        ori=0.0, pos=(-0.075625, .1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    dividing_line_3 = visual.Line(
        win=win, name='dividing_line_3',
        start=(-(0.1, 0.1)[0]/2.0, 0), end=(+(0.1, 0.1)[0]/2.0, 0),
        ori=90.0, pos=(0, .1), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor=[0.3255, 0.3255, 0.3255], fillColor=[0.3255, 0.3255, 0.3255],
        opacity=None, depth=-4.0, interpolate=True)
    text_11 = visual.TextStim(win=win, name='text_11',
        text='This symbol indicates that the left image is SOMEWHAT more likely than the right image to be tested. ',
        font='Open Sans',
        pos=(0, -.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    fill_rectangle_6 = visual.Rect(
        win=win, name='fill_rectangle_6',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, -.25), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-6.0, interpolate=True)
    right_cue3_2 = visual.Rect(
        win=win, name='right_cue3_2',
        width=(0.06875, 0.049)[0], height=(0.06875, 0.049)[1],
        ori=0.0, pos=(0.034375, -.25), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    left_cue3_2 = visual.Rect(
        win=win, name='left_cue3_2',
        width=(0.09625, 0.049)[0], height=(0.09625, 0.049)[1],
        ori=0.0, pos=(-0.048125, -.25), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    dividing_line_4 = visual.Line(
        win=win, name='dividing_line_4',
        start=(-(0.1, 0.1)[0]/2.0, 0), end=(+(0.1, 0.1)[0]/2.0, 0),
        ori=90.0, pos=(0, -.25), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor=[0.3255, 0.3255, 0.3255], fillColor=[0.3255, 0.3255, 0.3255],
        opacity=None, depth=-9.0, interpolate=True)
    key_resp_8 = keyboard.Keyboard(deviceName='key_resp_8')
    text_12 = visual.TextStim(win=win, name='text_12',
        text="[Press 'y' to continue]",
        font='Open Sans',
        pos=(0, -.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    
    # --- Initialize components for Routine "Instructions10" ---
    text_13 = visual.TextStim(win=win, name='text_13',
        text='This symbol indicates that the right image is MUCH more likely than the left image to be tested. ',
        font='Open Sans',
        pos=(0, .25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    fill_rectangle_7 = visual.Rect(
        win=win, name='fill_rectangle_7',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, .1), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    right_cue3_3 = visual.Rect(
        win=win, name='right_cue3_3',
        width=(0.15125, 0.049)[0], height=(0.15125, 0.049)[1],
        ori=0.0, pos=(0.075625, .1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    left_cue3_3 = visual.Rect(
        win=win, name='left_cue3_3',
        width=(0.01375, 0.049)[0], height=(0.01375, 0.049)[1],
        ori=0.0, pos=(-0.006875, .1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    dividing_line_6 = visual.Line(
        win=win, name='dividing_line_6',
        start=(-(0.1, 0.1)[0]/2.0, 0), end=(+(0.1, 0.1)[0]/2.0, 0),
        ori=90.0, pos=(0, .1), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor=[0.3255, 0.3255, 0.3255], fillColor=[0.3255, 0.3255, 0.3255],
        opacity=None, depth=-4.0, interpolate=True)
    text_14 = visual.TextStim(win=win, name='text_14',
        text='This symbol indicates that the right image is SOMEWHAT more likely than the left image to be tested. ',
        font='Open Sans',
        pos=(0, -.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    fill_rectangle_8 = visual.Rect(
        win=win, name='fill_rectangle_8',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, -.25), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-6.0, interpolate=True)
    right_cue3_4 = visual.Rect(
        win=win, name='right_cue3_4',
        width=(0.09625, 0.049)[0], height=(0.09625, 0.049)[1],
        ori=0.0, pos=(0.048125, -.25), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    left_cue3_4 = visual.Rect(
        win=win, name='left_cue3_4',
        width=(0.06875, 0.049)[0], height=(0.06875, 0.049)[1],
        ori=0.0, pos=(-0.034375, -.25), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    dividing_line_5 = visual.Line(
        win=win, name='dividing_line_5',
        start=(-(0.1, 0.1)[0]/2.0, 0), end=(+(0.1, 0.1)[0]/2.0, 0),
        ori=90.0, pos=(0, -.25), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor=[0.3255, 0.3255, 0.3255], fillColor=[0.3255, 0.3255, 0.3255],
        opacity=None, depth=-9.0, interpolate=True)
    key_resp_9 = keyboard.Keyboard(deviceName='key_resp_9')
    text_15 = visual.TextStim(win=win, name='text_15',
        text="[Press 'y' to continue]",
        font='Open Sans',
        pos=(0, -.35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    # Run 'Begin Experiment' code from code_16
    taskphases = 'taskphases_' + str(psydata) + '.csv'
    psydata += 1
    save_path = "subject_number.txt"
    psydata.saveAsWideText(save_path)
    
    
    # --- Initialize components for Routine "intro_txt" ---
    text_19 = visual.TextStim(win=win, name='text_19',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_11 = keyboard.Keyboard(deviceName='key_resp_11')
    
    # --- Initialize components for Routine "initial_wait" ---
    fixation_cross_4 = visual.TextStim(win=win, name='fixation_cross_4',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "memoranda" ---
    left_stim = visual.ImageStim(
        win=win,
        name='left_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-.35, 0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_stim = visual.ImageStim(
        win=win,
        name='right_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(.35,0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    fixation_cross_5 = visual.TextStim(win=win, name='fixation_cross_5',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "isi1" ---
    fixation_cross = visual.TextStim(win=win, name='fixation_cross',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "reliability_cue" ---
    fill_rectangle = visual.Rect(
        win=win, name='fill_rectangle',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    right_cue = visual.Rect(
        win=win, name='right_cue',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    left_cue = visual.Rect(
        win=win, name='left_cue',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    dividing_line = visual.Line(
        win=win, name='dividing_line',
        start=(-(0.1, 0.1)[0]/2.0, 0), end=(+(0.1, 0.1)[0]/2.0, 0),
        ori=90.0, pos=(0, 0), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor=[0.3255, 0.3255, 0.3255], fillColor=[0.3255, 0.3255, 0.3255],
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "isi2" ---
    fixation_cross2 = visual.TextStim(win=win, name='fixation_cross2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "ping" ---
    ping_im = visual.ImageStim(
        win=win,
        name='ping_im', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    fixation_cross2_2 = visual.TextStim(win=win, name='fixation_cross2_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    ping_im_2 = visual.ImageStim(
        win=win,
        name='ping_im_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    fixation_cross2_3 = visual.TextStim(win=win, name='fixation_cross2_3',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    ping_im_3 = visual.ImageStim(
        win=win,
        name='ping_im_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "noise_mask" ---
    noise_mask_im = visual.ImageStim(
        win=win,
        name='noise_mask_im', 
        image='stimuli/noise_mask.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.35, 0.35),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "isi3" ---
    fixation_cross_3 = visual.TextStim(win=win, name='fixation_cross_3',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "probe" ---
    picker = visual.Rect(
        win=win, name='picker',
        width=(0.26, 0.26)[0], height=(0.26, 0.26)[1],
        ori=0.0, pos=(2, 2), anchor='center',
        lineWidth=10.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    distractor_mem_stim = visual.ImageStim(
        win=win,
        name='distractor_mem_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    memoranda_stim = visual.ImageStim(
        win=win,
        name='memoranda_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    distractor_im_0 = visual.ImageStim(
        win=win,
        name='distractor_im_0', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    distractor_im_1 = visual.ImageStim(
        win=win,
        name='distractor_im_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    distractor_im_2 = visual.ImageStim(
        win=win,
        name='distractor_im_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    distractor_im_3 = visual.ImageStim(
        win=win,
        name='distractor_im_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    mem_cue = visual.ImageStim(
        win=win,
        name='mem_cue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    text_24 = visual.TextStim(win=win, name='text_24',
        text='',
        font='Open Sans',
        pos=(0, .09), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "block_break" ---
    break_txt = visual.TextStim(win=win, name='break_txt',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "ITI" ---
    fixation_iti = visual.TextStim(win=win, name='fixation_iti',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "repeat_practice_code" ---
    repeat_prac_text = visual.TextStim(win=win, name='repeat_prac_text',
        text="You made too many incorrect responses during the practice.\n\nPlease try the practice again before moving on to the main experiment.\n\nPress 'y' to continue.",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_12 = keyboard.Keyboard(deviceName='key_resp_12')
    
    # --- Initialize components for Routine "Instructions" ---
    Instructions_to_image_memory_test = visual.TextStim(win=win, name='Instructions_to_image_memory_test',
        text="This is the final part of the experiment. You will now see 240 images, each will be presented for 3 seconds and you need to click 1,2,3,4 to answer a question if you have seen the image overall in the experiment or not. 1 - Definitely have not seen, 2 - Probably have not seen, 3 - Probably have seen, 4 - Definitely have seen.\n\nOnce ready please click 'y'.",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_17 = keyboard.Keyboard(deviceName='key_resp_17')
    
    # --- Initialize components for Routine "image_recall_task" ---
    image_input = visual.ImageStim(
        win=win,
        name='image_input', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_recall_response = keyboard.Keyboard(deviceName='image_recall_response')
    
    # --- Initialize components for Routine "small_break" ---
    text_27 = visual.TextStim(win=win, name='text_27',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "break_2" ---
    break_txt_final = visual.TextStim(win=win, name='break_txt_final',
        text='Any text\n\nincluding line breaks',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_18 = keyboard.Keyboard(deviceName='key_resp_18')
    
    # --- Initialize components for Routine "end_exp" ---
    thanks = visual.TextStim(win=win, name='thanks',
        text='Thank you for your participation!\n\nThis page will redirect back to complete a short survey.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "setup_js" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('setup_js.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    setup_jsComponents = []
    for thisComponent in setup_jsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "setup_js" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in setup_jsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "setup_js" ---
    for thisComponent in setup_jsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('setup_js.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "setup_js" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instructions1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instructions1.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # keep track of which components have finished
    Instructions1Components = [text1, key_resp_2, text]
    for thisComponent in Instructions1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text1* updates
        
        # if text1 is starting this frame...
        if text1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text1.frameNStart = frameN  # exact frame index
            text1.tStart = t  # local t and not account for scr refresh
            text1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text1.started')
            # update status
            text1.status = STARTED
            text1.setAutoDraw(True)
        
        # if text1 is active this frame...
        if text1.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instructions1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions1" ---
    for thisComponent in Instructions1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instructions1.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Instructions1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Instructions2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions2.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_3
        key_resp_3.keys = []
        key_resp_3.rt = []
        _key_resp_3_allKeys = []
        # keep track of which components have finished
        Instructions2Components = [text_2, key_resp_3, text_3]
        for thisComponent in Instructions2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_3* updates
            waitOnFlip = False
            
            # if key_resp_3 is starting this frame...
            if key_resp_3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_3.frameNStart = frameN  # exact frame index
                key_resp_3.tStart = t  # local t and not account for scr refresh
                key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_3.started')
                # update status
                key_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_3.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_3_allKeys.extend(theseKeys)
                if len(_key_resp_3_allKeys):
                    key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                    key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                    key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions2" ---
        for thisComponent in Instructions2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions2.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_3.keys in ['', [], None]:  # No response was made
            key_resp_3.keys = None
        trials.addData('key_resp_3.keys',key_resp_3.keys)
        if key_resp_3.keys != None:  # we had a response
            trials.addData('key_resp_3.rt', key_resp_3.rt)
            trials.addData('key_resp_3.duration', key_resp_3.duration)
        # the Routine "Instructions2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions3.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_4
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        Instructions3Components = [text_4, key_resp_4, text_5]
        for thisComponent in Instructions3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_4* updates
            
            # if text_4 is starting this frame...
            if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.started')
                # update status
                text_4.status = STARTED
                text_4.setAutoDraw(True)
            
            # if text_4 is active this frame...
            if text_4.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.started')
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions3" ---
        for thisComponent in Instructions3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions3.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        trials.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            trials.addData('key_resp_4.rt', key_resp_4.rt)
            trials.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "Instructions3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions4" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions4.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_5
        key_resp_5.keys = []
        key_resp_5.rt = []
        _key_resp_5_allKeys = []
        # keep track of which components have finished
        Instructions4Components = [text_6, key_resp_5, text_7]
        for thisComponent in Instructions4Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions4" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.started')
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # *key_resp_5* updates
            waitOnFlip = False
            
            # if key_resp_5 is starting this frame...
            if key_resp_5.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_5.frameNStart = frameN  # exact frame index
                key_resp_5.tStart = t  # local t and not account for scr refresh
                key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_5.started')
                # update status
                key_resp_5.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_5.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_5.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_5_allKeys.extend(theseKeys)
                if len(_key_resp_5_allKeys):
                    key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                    key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                    key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_7* updates
            
            # if text_7 is starting this frame...
            if text_7.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_7.frameNStart = frameN  # exact frame index
                text_7.tStart = t  # local t and not account for scr refresh
                text_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_7.started')
                # update status
                text_7.status = STARTED
                text_7.setAutoDraw(True)
            
            # if text_7 is active this frame...
            if text_7.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions4Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions4" ---
        for thisComponent in Instructions4Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions4.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_5.keys in ['', [], None]:  # No response was made
            key_resp_5.keys = None
        trials.addData('key_resp_5.keys',key_resp_5.keys)
        if key_resp_5.keys != None:  # we had a response
            trials.addData('key_resp_5.rt', key_resp_5.rt)
            trials.addData('key_resp_5.duration', key_resp_5.duration)
        # the Routine "Instructions4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions5" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions5.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_10
        key_resp_10.keys = []
        key_resp_10.rt = []
        _key_resp_10_allKeys = []
        # keep track of which components have finished
        Instructions5Components = [text_16, image, text_17, image_2, text_18, key_resp_10]
        for thisComponent in Instructions5Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions5" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_16* updates
            
            # if text_16 is starting this frame...
            if text_16.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_16.frameNStart = frameN  # exact frame index
                text_16.tStart = t  # local t and not account for scr refresh
                text_16.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_16, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_16.started')
                # update status
                text_16.status = STARTED
                text_16.setAutoDraw(True)
            
            # if text_16 is active this frame...
            if text_16.status == STARTED:
                # update params
                pass
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # *text_17* updates
            
            # if text_17 is starting this frame...
            if text_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_17.frameNStart = frameN  # exact frame index
                text_17.tStart = t  # local t and not account for scr refresh
                text_17.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_17, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_17.started')
                # update status
                text_17.status = STARTED
                text_17.setAutoDraw(True)
            
            # if text_17 is active this frame...
            if text_17.status == STARTED:
                # update params
                pass
            
            # *image_2* updates
            
            # if image_2 is starting this frame...
            if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_2.frameNStart = frameN  # exact frame index
                image_2.tStart = t  # local t and not account for scr refresh
                image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.started')
                # update status
                image_2.status = STARTED
                image_2.setAutoDraw(True)
            
            # if image_2 is active this frame...
            if image_2.status == STARTED:
                # update params
                pass
            
            # *text_18* updates
            
            # if text_18 is starting this frame...
            if text_18.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_18.frameNStart = frameN  # exact frame index
                text_18.tStart = t  # local t and not account for scr refresh
                text_18.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_18, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_18.started')
                # update status
                text_18.status = STARTED
                text_18.setAutoDraw(True)
            
            # if text_18 is active this frame...
            if text_18.status == STARTED:
                # update params
                pass
            
            # *key_resp_10* updates
            waitOnFlip = False
            
            # if key_resp_10 is starting this frame...
            if key_resp_10.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_10.frameNStart = frameN  # exact frame index
                key_resp_10.tStart = t  # local t and not account for scr refresh
                key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_10.started')
                # update status
                key_resp_10.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_10.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_10.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_10_allKeys.extend(theseKeys)
                if len(_key_resp_10_allKeys):
                    key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                    key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                    key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions5Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions5" ---
        for thisComponent in Instructions5Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions5.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_10.keys in ['', [], None]:  # No response was made
            key_resp_10.keys = None
        trials.addData('key_resp_10.keys',key_resp_10.keys)
        if key_resp_10.keys != None:  # we had a response
            trials.addData('key_resp_10.rt', key_resp_10.rt)
            trials.addData('key_resp_10.duration', key_resp_10.duration)
        # the Routine "Instructions5" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions6b" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions6b.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_16
        key_resp_16.keys = []
        key_resp_16.rt = []
        _key_resp_16_allKeys = []
        # keep track of which components have finished
        Instructions6bComponents = [text_25, key_resp_16, text_26]
        for thisComponent in Instructions6bComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions6b" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_25* updates
            
            # if text_25 is starting this frame...
            if text_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_25.frameNStart = frameN  # exact frame index
                text_25.tStart = t  # local t and not account for scr refresh
                text_25.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_25, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_25.started')
                # update status
                text_25.status = STARTED
                text_25.setAutoDraw(True)
            
            # if text_25 is active this frame...
            if text_25.status == STARTED:
                # update params
                pass
            
            # *key_resp_16* updates
            waitOnFlip = False
            
            # if key_resp_16 is starting this frame...
            if key_resp_16.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_16.frameNStart = frameN  # exact frame index
                key_resp_16.tStart = t  # local t and not account for scr refresh
                key_resp_16.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_16, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_16.started')
                # update status
                key_resp_16.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_16.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_16.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_16.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_16.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_16_allKeys.extend(theseKeys)
                if len(_key_resp_16_allKeys):
                    key_resp_16.keys = _key_resp_16_allKeys[-1].name  # just the last key pressed
                    key_resp_16.rt = _key_resp_16_allKeys[-1].rt
                    key_resp_16.duration = _key_resp_16_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_26* updates
            
            # if text_26 is starting this frame...
            if text_26.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_26.frameNStart = frameN  # exact frame index
                text_26.tStart = t  # local t and not account for scr refresh
                text_26.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_26, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_26.started')
                # update status
                text_26.status = STARTED
                text_26.setAutoDraw(True)
            
            # if text_26 is active this frame...
            if text_26.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions6bComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions6b" ---
        for thisComponent in Instructions6bComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions6b.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_16.keys in ['', [], None]:  # No response was made
            key_resp_16.keys = None
        trials.addData('key_resp_16.keys',key_resp_16.keys)
        if key_resp_16.keys != None:  # we had a response
            trials.addData('key_resp_16.rt', key_resp_16.rt)
            trials.addData('key_resp_16.duration', key_resp_16.duration)
        # the Routine "Instructions6b" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        repeat_prac = data.TrialHandler(nReps=5.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='repeat_prac')
        thisExp.addLoop(repeat_prac)  # add the loop to the experiment
        thisRepeat_prac = repeat_prac.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRepeat_prac.rgb)
        if thisRepeat_prac != None:
            for paramName in thisRepeat_prac:
                globals()[paramName] = thisRepeat_prac[paramName]
        
        for thisRepeat_prac in repeat_prac:
            currentLoop = repeat_prac
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisRepeat_prac.rgb)
            if thisRepeat_prac != None:
                for paramName in thisRepeat_prac:
                    globals()[paramName] = thisRepeat_prac[paramName]
            
            # --- Prepare to start Routine "Instructions6" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Instructions6.started', globalClock.getTime(format='float'))
            # create starting attributes for key_resp_13
            key_resp_13.keys = []
            key_resp_13.rt = []
            _key_resp_13_allKeys = []
            # keep track of which components have finished
            Instructions6Components = [text_20, key_resp_13, text_21]
            for thisComponent in Instructions6Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Instructions6" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_20* updates
                
                # if text_20 is starting this frame...
                if text_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_20.frameNStart = frameN  # exact frame index
                    text_20.tStart = t  # local t and not account for scr refresh
                    text_20.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_20, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_20.started')
                    # update status
                    text_20.status = STARTED
                    text_20.setAutoDraw(True)
                
                # if text_20 is active this frame...
                if text_20.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_13* updates
                waitOnFlip = False
                
                # if key_resp_13 is starting this frame...
                if key_resp_13.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_13.frameNStart = frameN  # exact frame index
                    key_resp_13.tStart = t  # local t and not account for scr refresh
                    key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_13.started')
                    # update status
                    key_resp_13.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_13.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_13.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_13_allKeys.extend(theseKeys)
                    if len(_key_resp_13_allKeys):
                        key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                        key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                        key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *text_21* updates
                
                # if text_21 is starting this frame...
                if text_21.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    text_21.frameNStart = frameN  # exact frame index
                    text_21.tStart = t  # local t and not account for scr refresh
                    text_21.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_21, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_21.started')
                    # update status
                    text_21.status = STARTED
                    text_21.setAutoDraw(True)
                
                # if text_21 is active this frame...
                if text_21.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Instructions6Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Instructions6" ---
            for thisComponent in Instructions6Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Instructions6.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_resp_13.keys in ['', [], None]:  # No response was made
                key_resp_13.keys = None
            repeat_prac.addData('key_resp_13.keys',key_resp_13.keys)
            if key_resp_13.keys != None:  # we had a response
                repeat_prac.addData('key_resp_13.rt', key_resp_13.rt)
                repeat_prac.addData('key_resp_13.duration', key_resp_13.duration)
            # the Routine "Instructions6" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            prac_trials = data.TrialHandler(nReps=1.0, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions('initial_practice.csv'),
                seed=None, name='prac_trials')
            thisExp.addLoop(prac_trials)  # add the loop to the experiment
            thisPrac_trial = prac_trials.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_trial.rgb)
            if thisPrac_trial != None:
                for paramName in thisPrac_trial:
                    globals()[paramName] = thisPrac_trial[paramName]
            
            for thisPrac_trial in prac_trials:
                currentLoop = prac_trials
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisPrac_trial.rgb)
                if thisPrac_trial != None:
                    for paramName in thisPrac_trial:
                        globals()[paramName] = thisPrac_trial[paramName]
                
                # --- Prepare to start Routine "memoranda" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('memoranda.started', globalClock.getTime(format='float'))
                left_stim.setImage(left_img)
                right_stim.setImage(right_img)
                # keep track of which components have finished
                memorandaComponents = [left_stim, right_stim, fixation_cross_5]
                for thisComponent in memorandaComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "memoranda" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *left_stim* updates
                    
                    # if left_stim is starting this frame...
                    if left_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        left_stim.frameNStart = frameN  # exact frame index
                        left_stim.tStart = t  # local t and not account for scr refresh
                        left_stim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(left_stim, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'left_stim.started')
                        # update status
                        left_stim.status = STARTED
                        left_stim.setAutoDraw(True)
                    
                    # if left_stim is active this frame...
                    if left_stim.status == STARTED:
                        # update params
                        pass
                    
                    # if left_stim is stopping this frame...
                    if left_stim.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > left_stim.tStartRefresh + cue_time-frameTolerance:
                            # keep track of stop time/frame for later
                            left_stim.tStop = t  # not accounting for scr refresh
                            left_stim.tStopRefresh = tThisFlipGlobal  # on global time
                            left_stim.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'left_stim.stopped')
                            # update status
                            left_stim.status = FINISHED
                            left_stim.setAutoDraw(False)
                    
                    # *right_stim* updates
                    
                    # if right_stim is starting this frame...
                    if right_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        right_stim.frameNStart = frameN  # exact frame index
                        right_stim.tStart = t  # local t and not account for scr refresh
                        right_stim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(right_stim, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'right_stim.started')
                        # update status
                        right_stim.status = STARTED
                        right_stim.setAutoDraw(True)
                    
                    # if right_stim is active this frame...
                    if right_stim.status == STARTED:
                        # update params
                        pass
                    
                    # if right_stim is stopping this frame...
                    if right_stim.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > right_stim.tStartRefresh + cue_time-frameTolerance:
                            # keep track of stop time/frame for later
                            right_stim.tStop = t  # not accounting for scr refresh
                            right_stim.tStopRefresh = tThisFlipGlobal  # on global time
                            right_stim.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'right_stim.stopped')
                            # update status
                            right_stim.status = FINISHED
                            right_stim.setAutoDraw(False)
                    
                    # *fixation_cross_5* updates
                    
                    # if fixation_cross_5 is starting this frame...
                    if fixation_cross_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        fixation_cross_5.frameNStart = frameN  # exact frame index
                        fixation_cross_5.tStart = t  # local t and not account for scr refresh
                        fixation_cross_5.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(fixation_cross_5, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_cross_5.started')
                        # update status
                        fixation_cross_5.status = STARTED
                        fixation_cross_5.setAutoDraw(True)
                    
                    # if fixation_cross_5 is active this frame...
                    if fixation_cross_5.status == STARTED:
                        # update params
                        pass
                    
                    # if fixation_cross_5 is stopping this frame...
                    if fixation_cross_5.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > fixation_cross_5.tStartRefresh + cue_time-frameTolerance:
                            # keep track of stop time/frame for later
                            fixation_cross_5.tStop = t  # not accounting for scr refresh
                            fixation_cross_5.tStopRefresh = tThisFlipGlobal  # on global time
                            fixation_cross_5.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'fixation_cross_5.stopped')
                            # update status
                            fixation_cross_5.status = FINISHED
                            fixation_cross_5.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in memorandaComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "memoranda" ---
                for thisComponent in memorandaComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('memoranda.stopped', globalClock.getTime(format='float'))
                # the Routine "memoranda" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "isi1" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('isi1.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                isi1Components = [fixation_cross]
                for thisComponent in isi1Components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "isi1" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *fixation_cross* updates
                    
                    # if fixation_cross is starting this frame...
                    if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        fixation_cross.frameNStart = frameN  # exact frame index
                        fixation_cross.tStart = t  # local t and not account for scr refresh
                        fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_cross.started')
                        # update status
                        fixation_cross.status = STARTED
                        fixation_cross.setAutoDraw(True)
                    
                    # if fixation_cross is active this frame...
                    if fixation_cross.status == STARTED:
                        # update params
                        pass
                    
                    # if fixation_cross is stopping this frame...
                    if fixation_cross.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > fixation_cross.tStartRefresh + delay_1_time-frameTolerance:
                            # keep track of stop time/frame for later
                            fixation_cross.tStop = t  # not accounting for scr refresh
                            fixation_cross.tStopRefresh = tThisFlipGlobal  # on global time
                            fixation_cross.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'fixation_cross.stopped')
                            # update status
                            fixation_cross.status = FINISHED
                            fixation_cross.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in isi1Components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "isi1" ---
                for thisComponent in isi1Components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('isi1.stopped', globalClock.getTime(format='float'))
                # the Routine "isi1" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "probe" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('probe.started', globalClock.getTime(format='float'))
                distractor_mem_stim.setPos((dismem_x, dismem_y))
                distractor_mem_stim.setImage(distractor_stim)
                memoranda_stim.setPos((mem_x, mem_y))
                memoranda_stim.setImage(mem_stim)
                distractor_im_0.setPos((dis0_x, dis0_y))
                distractor_im_0.setImage(distractor_0)
                distractor_im_1.setPos((dis1_x, dis1_y))
                distractor_im_1.setImage(distractor_1)
                distractor_im_2.setPos((dis2_x, dis2_y))
                distractor_im_2.setImage(distractor_2)
                distractor_im_3.setPos((dis3_x, dis3_y))
                distractor_im_3.setImage(distractor_3)
                # setup some python lists for storing info about the mouse
                mouse.x = []
                mouse.y = []
                mouse.leftButton = []
                mouse.midButton = []
                mouse.rightButton = []
                mouse.time = []
                mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                mem_cue.setImage(cue_img)
                text_24.setText('instruct_guide_txt')
                # Run 'Begin Routine' code from code_2
                
                // inserted the crib sheet stuff
                thisExp=psychoJS.experiment;
                win=psychoJS.window;
                event=psychoJS.eventManager;
                
                
                //hide mouse
                document.body.style.cursor='none'
                
                //exit if on mobile
                if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
                  quitPsychoJS('Mobile device detected. Goodbye!', false)
                }
                # keep track of which components have finished
                probeComponents = [picker, distractor_mem_stim, memoranda_stim, distractor_im_0, distractor_im_1, distractor_im_2, distractor_im_3, mouse, mem_cue, text_24]
                for thisComponent in probeComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "probe" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *picker* updates
                    
                    # if picker is starting this frame...
                    if picker.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        picker.frameNStart = frameN  # exact frame index
                        picker.tStart = t  # local t and not account for scr refresh
                        picker.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(picker, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'picker.started')
                        # update status
                        picker.status = STARTED
                        picker.setAutoDraw(True)
                    
                    # if picker is active this frame...
                    if picker.status == STARTED:
                        # update params
                        pass
                    
                    # if picker is stopping this frame...
                    if picker.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > picker.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            picker.tStop = t  # not accounting for scr refresh
                            picker.tStopRefresh = tThisFlipGlobal  # on global time
                            picker.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'picker.stopped')
                            # update status
                            picker.status = FINISHED
                            picker.setAutoDraw(False)
                    
                    # *distractor_mem_stim* updates
                    
                    # if distractor_mem_stim is starting this frame...
                    if distractor_mem_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_mem_stim.frameNStart = frameN  # exact frame index
                        distractor_mem_stim.tStart = t  # local t and not account for scr refresh
                        distractor_mem_stim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_mem_stim, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_mem_stim.started')
                        # update status
                        distractor_mem_stim.status = STARTED
                        distractor_mem_stim.setAutoDraw(True)
                    
                    # if distractor_mem_stim is active this frame...
                    if distractor_mem_stim.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_mem_stim is stopping this frame...
                    if distractor_mem_stim.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_mem_stim.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_mem_stim.tStop = t  # not accounting for scr refresh
                            distractor_mem_stim.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_mem_stim.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_mem_stim.stopped')
                            # update status
                            distractor_mem_stim.status = FINISHED
                            distractor_mem_stim.setAutoDraw(False)
                    
                    # *memoranda_stim* updates
                    
                    # if memoranda_stim is starting this frame...
                    if memoranda_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        memoranda_stim.frameNStart = frameN  # exact frame index
                        memoranda_stim.tStart = t  # local t and not account for scr refresh
                        memoranda_stim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(memoranda_stim, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'memoranda_stim.started')
                        # update status
                        memoranda_stim.status = STARTED
                        memoranda_stim.setAutoDraw(True)
                    
                    # if memoranda_stim is active this frame...
                    if memoranda_stim.status == STARTED:
                        # update params
                        pass
                    
                    # if memoranda_stim is stopping this frame...
                    if memoranda_stim.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > memoranda_stim.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            memoranda_stim.tStop = t  # not accounting for scr refresh
                            memoranda_stim.tStopRefresh = tThisFlipGlobal  # on global time
                            memoranda_stim.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'memoranda_stim.stopped')
                            # update status
                            memoranda_stim.status = FINISHED
                            memoranda_stim.setAutoDraw(False)
                    
                    # *distractor_im_0* updates
                    
                    # if distractor_im_0 is starting this frame...
                    if distractor_im_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_0.frameNStart = frameN  # exact frame index
                        distractor_im_0.tStart = t  # local t and not account for scr refresh
                        distractor_im_0.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_0, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_0.started')
                        # update status
                        distractor_im_0.status = STARTED
                        distractor_im_0.setAutoDraw(True)
                    
                    # if distractor_im_0 is active this frame...
                    if distractor_im_0.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_0 is stopping this frame...
                    if distractor_im_0.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_0.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_0.tStop = t  # not accounting for scr refresh
                            distractor_im_0.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_0.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_0.stopped')
                            # update status
                            distractor_im_0.status = FINISHED
                            distractor_im_0.setAutoDraw(False)
                    
                    # *distractor_im_1* updates
                    
                    # if distractor_im_1 is starting this frame...
                    if distractor_im_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_1.frameNStart = frameN  # exact frame index
                        distractor_im_1.tStart = t  # local t and not account for scr refresh
                        distractor_im_1.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_1, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_1.started')
                        # update status
                        distractor_im_1.status = STARTED
                        distractor_im_1.setAutoDraw(True)
                    
                    # if distractor_im_1 is active this frame...
                    if distractor_im_1.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_1 is stopping this frame...
                    if distractor_im_1.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_1.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_1.tStop = t  # not accounting for scr refresh
                            distractor_im_1.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_1.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_1.stopped')
                            # update status
                            distractor_im_1.status = FINISHED
                            distractor_im_1.setAutoDraw(False)
                    
                    # *distractor_im_2* updates
                    
                    # if distractor_im_2 is starting this frame...
                    if distractor_im_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_2.frameNStart = frameN  # exact frame index
                        distractor_im_2.tStart = t  # local t and not account for scr refresh
                        distractor_im_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_2.started')
                        # update status
                        distractor_im_2.status = STARTED
                        distractor_im_2.setAutoDraw(True)
                    
                    # if distractor_im_2 is active this frame...
                    if distractor_im_2.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_2 is stopping this frame...
                    if distractor_im_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_2.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_2.tStop = t  # not accounting for scr refresh
                            distractor_im_2.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_2.stopped')
                            # update status
                            distractor_im_2.status = FINISHED
                            distractor_im_2.setAutoDraw(False)
                    
                    # *distractor_im_3* updates
                    
                    # if distractor_im_3 is starting this frame...
                    if distractor_im_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_3.frameNStart = frameN  # exact frame index
                        distractor_im_3.tStart = t  # local t and not account for scr refresh
                        distractor_im_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_3, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_3.started')
                        # update status
                        distractor_im_3.status = STARTED
                        distractor_im_3.setAutoDraw(True)
                    
                    # if distractor_im_3 is active this frame...
                    if distractor_im_3.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_3 is stopping this frame...
                    if distractor_im_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_3.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_3.tStop = t  # not accounting for scr refresh
                            distractor_im_3.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_3.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_3.stopped')
                            # update status
                            distractor_im_3.status = FINISHED
                            distractor_im_3.setAutoDraw(False)
                    # *mouse* updates
                    
                    # if mouse is starting this frame...
                    if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mouse.frameNStart = frameN  # exact frame index
                        mouse.tStart = t  # local t and not account for scr refresh
                        mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse.started', t)
                        # update status
                        mouse.status = STARTED
                        mouse.mouseClock.reset()
                        prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse is stopping this frame...
                    if mouse.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse.tStop = t  # not accounting for scr refresh
                            mouse.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse.stopped', t)
                            # update status
                            mouse.status = FINISHED
                    if mouse.status == STARTED:  # only update if started and not finished!
                        buttons = mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([memoranda_stim, distractor_mem_stim, distractor_im_0, distractor_im_1, distractor_im_2, distractor_im_3], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse):
                                        gotValidClick = True
                                        mouse.clicked_name.append(obj.name)
                                x, y = mouse.getPos()
                                mouse.x.append(x)
                                mouse.y.append(y)
                                buttons = mouse.getPressed()
                                mouse.leftButton.append(buttons[0])
                                mouse.midButton.append(buttons[1])
                                mouse.rightButton.append(buttons[2])
                                mouse.time.append(mouse.mouseClock.getTime())
                    
                    # *mem_cue* updates
                    
                    # if mem_cue is starting this frame...
                    if mem_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mem_cue.frameNStart = frameN  # exact frame index
                        mem_cue.tStart = t  # local t and not account for scr refresh
                        mem_cue.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mem_cue, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'mem_cue.started')
                        # update status
                        mem_cue.status = STARTED
                        mem_cue.setAutoDraw(True)
                    
                    # if mem_cue is active this frame...
                    if mem_cue.status == STARTED:
                        # update params
                        pass
                    
                    # if mem_cue is stopping this frame...
                    if mem_cue.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mem_cue.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            mem_cue.tStop = t  # not accounting for scr refresh
                            mem_cue.tStopRefresh = tThisFlipGlobal  # on global time
                            mem_cue.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'mem_cue.stopped')
                            # update status
                            mem_cue.status = FINISHED
                            mem_cue.setAutoDraw(False)
                    
                    # *text_24* updates
                    
                    # if text_24 is starting this frame...
                    if text_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_24.frameNStart = frameN  # exact frame index
                        text_24.tStart = t  # local t and not account for scr refresh
                        text_24.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_24, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_24.started')
                        # update status
                        text_24.status = STARTED
                        text_24.setAutoDraw(True)
                    
                    # if text_24 is active this frame...
                    if text_24.status == STARTED:
                        # update params
                        pass
                    
                    # if text_24 is stopping this frame...
                    if text_24.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_24.tStartRefresh + probe_time-frameTolerance:
                            # keep track of stop time/frame for later
                            text_24.tStop = t  # not accounting for scr refresh
                            text_24.tStopRefresh = tThisFlipGlobal  # on global time
                            text_24.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_24.stopped')
                            # update status
                            text_24.status = FINISHED
                            text_24.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in probeComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "probe" ---
                for thisComponent in probeComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('probe.stopped', globalClock.getTime(format='float'))
                # store data for prac_trials (TrialHandler)
                prac_trials.addData('mouse.x', mouse.x)
                prac_trials.addData('mouse.y', mouse.y)
                prac_trials.addData('mouse.leftButton', mouse.leftButton)
                prac_trials.addData('mouse.midButton', mouse.midButton)
                prac_trials.addData('mouse.rightButton', mouse.rightButton)
                prac_trials.addData('mouse.time', mouse.time)
                prac_trials.addData('mouse.clicked_name', mouse.clicked_name)
                # the Routine "probe" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "correct_ans" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('correct_ans.started', globalClock.getTime(format='float'))
                correct_border.setPos((mem_x, mem_y))
                distractor_mem_stim_3.setPos((dismem_x, dismem_y))
                distractor_mem_stim_3.setImage(distractor_stim)
                memoranda_stim_2.setPos((mem_x, mem_y))
                memoranda_stim_2.setImage(mem_stim)
                distractor_im.setPos((dis0_x, dis0_y))
                distractor_im.setImage(distractor_0)
                distractor_im_4.setPos((dis1_x, dis1_y))
                distractor_im_4.setImage(distractor_1)
                distractor_im_5.setPos((dis2_x, dis2_y))
                distractor_im_5.setImage(distractor_2)
                distractor_im_6.setPos((dis2_x, dis2_y))
                distractor_im_6.setImage(distractor_2)
                distractor_im_7.setPos((dis3_x, dis3_y))
                distractor_im_7.setImage(distractor_3)
                mem_cue_2.setImage(cue_img)
                # keep track of which components have finished
                correct_ansComponents = [correct_border, distractor_mem_stim_3, memoranda_stim_2, distractor_im, distractor_im_4, distractor_im_5, distractor_im_6, distractor_im_7, mem_cue_2]
                for thisComponent in correct_ansComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "correct_ans" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *correct_border* updates
                    
                    # if correct_border is starting this frame...
                    if correct_border.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        correct_border.frameNStart = frameN  # exact frame index
                        correct_border.tStart = t  # local t and not account for scr refresh
                        correct_border.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(correct_border, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'correct_border.started')
                        # update status
                        correct_border.status = STARTED
                        correct_border.setAutoDraw(True)
                    
                    # if correct_border is active this frame...
                    if correct_border.status == STARTED:
                        # update params
                        pass
                    
                    # if correct_border is stopping this frame...
                    if correct_border.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > correct_border.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            correct_border.tStop = t  # not accounting for scr refresh
                            correct_border.tStopRefresh = tThisFlipGlobal  # on global time
                            correct_border.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'correct_border.stopped')
                            # update status
                            correct_border.status = FINISHED
                            correct_border.setAutoDraw(False)
                    
                    # *distractor_mem_stim_3* updates
                    
                    # if distractor_mem_stim_3 is starting this frame...
                    if distractor_mem_stim_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_mem_stim_3.frameNStart = frameN  # exact frame index
                        distractor_mem_stim_3.tStart = t  # local t and not account for scr refresh
                        distractor_mem_stim_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_mem_stim_3, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_mem_stim_3.started')
                        # update status
                        distractor_mem_stim_3.status = STARTED
                        distractor_mem_stim_3.setAutoDraw(True)
                    
                    # if distractor_mem_stim_3 is active this frame...
                    if distractor_mem_stim_3.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_mem_stim_3 is stopping this frame...
                    if distractor_mem_stim_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_mem_stim_3.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_mem_stim_3.tStop = t  # not accounting for scr refresh
                            distractor_mem_stim_3.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_mem_stim_3.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_mem_stim_3.stopped')
                            # update status
                            distractor_mem_stim_3.status = FINISHED
                            distractor_mem_stim_3.setAutoDraw(False)
                    
                    # *memoranda_stim_2* updates
                    
                    # if memoranda_stim_2 is starting this frame...
                    if memoranda_stim_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        memoranda_stim_2.frameNStart = frameN  # exact frame index
                        memoranda_stim_2.tStart = t  # local t and not account for scr refresh
                        memoranda_stim_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(memoranda_stim_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'memoranda_stim_2.started')
                        # update status
                        memoranda_stim_2.status = STARTED
                        memoranda_stim_2.setAutoDraw(True)
                    
                    # if memoranda_stim_2 is active this frame...
                    if memoranda_stim_2.status == STARTED:
                        # update params
                        pass
                    
                    # if memoranda_stim_2 is stopping this frame...
                    if memoranda_stim_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > memoranda_stim_2.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            memoranda_stim_2.tStop = t  # not accounting for scr refresh
                            memoranda_stim_2.tStopRefresh = tThisFlipGlobal  # on global time
                            memoranda_stim_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'memoranda_stim_2.stopped')
                            # update status
                            memoranda_stim_2.status = FINISHED
                            memoranda_stim_2.setAutoDraw(False)
                    
                    # *distractor_im* updates
                    
                    # if distractor_im is starting this frame...
                    if distractor_im.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im.frameNStart = frameN  # exact frame index
                        distractor_im.tStart = t  # local t and not account for scr refresh
                        distractor_im.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im.started')
                        # update status
                        distractor_im.status = STARTED
                        distractor_im.setAutoDraw(True)
                    
                    # if distractor_im is active this frame...
                    if distractor_im.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im is stopping this frame...
                    if distractor_im.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im.tStop = t  # not accounting for scr refresh
                            distractor_im.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im.stopped')
                            # update status
                            distractor_im.status = FINISHED
                            distractor_im.setAutoDraw(False)
                    
                    # *distractor_im_4* updates
                    
                    # if distractor_im_4 is starting this frame...
                    if distractor_im_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_4.frameNStart = frameN  # exact frame index
                        distractor_im_4.tStart = t  # local t and not account for scr refresh
                        distractor_im_4.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_4, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_4.started')
                        # update status
                        distractor_im_4.status = STARTED
                        distractor_im_4.setAutoDraw(True)
                    
                    # if distractor_im_4 is active this frame...
                    if distractor_im_4.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_4 is stopping this frame...
                    if distractor_im_4.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_4.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_4.tStop = t  # not accounting for scr refresh
                            distractor_im_4.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_4.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_4.stopped')
                            # update status
                            distractor_im_4.status = FINISHED
                            distractor_im_4.setAutoDraw(False)
                    
                    # *distractor_im_5* updates
                    
                    # if distractor_im_5 is starting this frame...
                    if distractor_im_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_5.frameNStart = frameN  # exact frame index
                        distractor_im_5.tStart = t  # local t and not account for scr refresh
                        distractor_im_5.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_5, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_5.started')
                        # update status
                        distractor_im_5.status = STARTED
                        distractor_im_5.setAutoDraw(True)
                    
                    # if distractor_im_5 is active this frame...
                    if distractor_im_5.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_5 is stopping this frame...
                    if distractor_im_5.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_5.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_5.tStop = t  # not accounting for scr refresh
                            distractor_im_5.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_5.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_5.stopped')
                            # update status
                            distractor_im_5.status = FINISHED
                            distractor_im_5.setAutoDraw(False)
                    
                    # *distractor_im_6* updates
                    
                    # if distractor_im_6 is starting this frame...
                    if distractor_im_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_6.frameNStart = frameN  # exact frame index
                        distractor_im_6.tStart = t  # local t and not account for scr refresh
                        distractor_im_6.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_6, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_6.started')
                        # update status
                        distractor_im_6.status = STARTED
                        distractor_im_6.setAutoDraw(True)
                    
                    # if distractor_im_6 is active this frame...
                    if distractor_im_6.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_6 is stopping this frame...
                    if distractor_im_6.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_6.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_6.tStop = t  # not accounting for scr refresh
                            distractor_im_6.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_6.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_6.stopped')
                            # update status
                            distractor_im_6.status = FINISHED
                            distractor_im_6.setAutoDraw(False)
                    
                    # *distractor_im_7* updates
                    
                    # if distractor_im_7 is starting this frame...
                    if distractor_im_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        distractor_im_7.frameNStart = frameN  # exact frame index
                        distractor_im_7.tStart = t  # local t and not account for scr refresh
                        distractor_im_7.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(distractor_im_7, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'distractor_im_7.started')
                        # update status
                        distractor_im_7.status = STARTED
                        distractor_im_7.setAutoDraw(True)
                    
                    # if distractor_im_7 is active this frame...
                    if distractor_im_7.status == STARTED:
                        # update params
                        pass
                    
                    # if distractor_im_7 is stopping this frame...
                    if distractor_im_7.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > distractor_im_7.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            distractor_im_7.tStop = t  # not accounting for scr refresh
                            distractor_im_7.tStopRefresh = tThisFlipGlobal  # on global time
                            distractor_im_7.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'distractor_im_7.stopped')
                            # update status
                            distractor_im_7.status = FINISHED
                            distractor_im_7.setAutoDraw(False)
                    
                    # *mem_cue_2* updates
                    
                    # if mem_cue_2 is starting this frame...
                    if mem_cue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mem_cue_2.frameNStart = frameN  # exact frame index
                        mem_cue_2.tStart = t  # local t and not account for scr refresh
                        mem_cue_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mem_cue_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'mem_cue_2.started')
                        # update status
                        mem_cue_2.status = STARTED
                        mem_cue_2.setAutoDraw(True)
                    
                    # if mem_cue_2 is active this frame...
                    if mem_cue_2.status == STARTED:
                        # update params
                        pass
                    
                    # if mem_cue_2 is stopping this frame...
                    if mem_cue_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mem_cue_2.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            mem_cue_2.tStop = t  # not accounting for scr refresh
                            mem_cue_2.tStopRefresh = tThisFlipGlobal  # on global time
                            mem_cue_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'mem_cue_2.stopped')
                            # update status
                            mem_cue_2.status = FINISHED
                            mem_cue_2.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in correct_ansComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "correct_ans" ---
                for thisComponent in correct_ansComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('correct_ans.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.000000)
                
                # --- Prepare to start Routine "ITI" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('ITI.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                ITIComponents = [fixation_iti]
                for thisComponent in ITIComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "ITI" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *fixation_iti* updates
                    
                    # if fixation_iti is starting this frame...
                    if fixation_iti.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        fixation_iti.frameNStart = frameN  # exact frame index
                        fixation_iti.tStart = t  # local t and not account for scr refresh
                        fixation_iti.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(fixation_iti, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_iti.started')
                        # update status
                        fixation_iti.status = STARTED
                        fixation_iti.setAutoDraw(True)
                    
                    # if fixation_iti is active this frame...
                    if fixation_iti.status == STARTED:
                        # update params
                        pass
                    
                    # if fixation_iti is stopping this frame...
                    if fixation_iti.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > fixation_iti.tStartRefresh + iti_dur-frameTolerance:
                            # keep track of stop time/frame for later
                            fixation_iti.tStop = t  # not accounting for scr refresh
                            fixation_iti.tStopRefresh = tThisFlipGlobal  # on global time
                            fixation_iti.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'fixation_iti.stopped')
                            # update status
                            fixation_iti.status = FINISHED
                            fixation_iti.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in ITIComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "ITI" ---
                for thisComponent in ITIComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('ITI.stopped', globalClock.getTime(format='float'))
                # the Routine "ITI" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 1.0 repeats of 'prac_trials'
            
            
            # --- Prepare to start Routine "repeat_code" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('repeat_code.started', globalClock.getTime(format='float'))
            # create starting attributes for key_resp_14
            key_resp_14.keys = []
            key_resp_14.rt = []
            _key_resp_14_allKeys = []
            # Run 'Begin Routine' code from code_6
            mean_error = task_errors.reduce((a, b) => a + b) / task_errors.length
            
            console.log('\nepoch summaries\n')
            console.log(task_errors)
            console.log(task_errors.length)
            console.log(mean_error)
            
            //check if practice and mean error is over a threshold
            if (mean_error > 45 && taskPhase == 'practice') {
                console.log('repeating practice');
                task_errors = [];
                }
            else {
                continueRoutine = false; 
                repeat_practice.finished = true;
            };
            # keep track of which components have finished
            repeat_codeComponents = [repeat_prac_text_2, key_resp_14]
            for thisComponent in repeat_codeComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "repeat_code" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *repeat_prac_text_2* updates
                
                # if repeat_prac_text_2 is starting this frame...
                if repeat_prac_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    repeat_prac_text_2.frameNStart = frameN  # exact frame index
                    repeat_prac_text_2.tStart = t  # local t and not account for scr refresh
                    repeat_prac_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(repeat_prac_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'repeat_prac_text_2.started')
                    # update status
                    repeat_prac_text_2.status = STARTED
                    repeat_prac_text_2.setAutoDraw(True)
                
                # if repeat_prac_text_2 is active this frame...
                if repeat_prac_text_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_14* updates
                waitOnFlip = False
                
                # if key_resp_14 is starting this frame...
                if key_resp_14.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_14.frameNStart = frameN  # exact frame index
                    key_resp_14.tStart = t  # local t and not account for scr refresh
                    key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_14.started')
                    # update status
                    key_resp_14.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_14.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_14.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_14_allKeys.extend(theseKeys)
                    if len(_key_resp_14_allKeys):
                        key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                        key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                        key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in repeat_codeComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "repeat_code" ---
            for thisComponent in repeat_codeComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('repeat_code.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_resp_14.keys in ['', [], None]:  # No response was made
                key_resp_14.keys = None
            repeat_prac.addData('key_resp_14.keys',key_resp_14.keys)
            if key_resp_14.keys != None:  # we had a response
                repeat_prac.addData('key_resp_14.rt', key_resp_14.rt)
                repeat_prac.addData('key_resp_14.duration', key_resp_14.duration)
            # the Routine "repeat_code" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 5.0 repeats of 'repeat_prac'
        
        
        # --- Prepare to start Routine "Instructions7" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions7.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_15
        key_resp_15.keys = []
        key_resp_15.rt = []
        _key_resp_15_allKeys = []
        # keep track of which components have finished
        Instructions7Components = [text_22, key_resp_15, text_23]
        for thisComponent in Instructions7Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions7" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_22* updates
            
            # if text_22 is starting this frame...
            if text_22.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_22.frameNStart = frameN  # exact frame index
                text_22.tStart = t  # local t and not account for scr refresh
                text_22.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_22, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_22.started')
                # update status
                text_22.status = STARTED
                text_22.setAutoDraw(True)
            
            # if text_22 is active this frame...
            if text_22.status == STARTED:
                # update params
                pass
            
            # *key_resp_15* updates
            waitOnFlip = False
            
            # if key_resp_15 is starting this frame...
            if key_resp_15.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_15.frameNStart = frameN  # exact frame index
                key_resp_15.tStart = t  # local t and not account for scr refresh
                key_resp_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_15.started')
                # update status
                key_resp_15.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_15.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_15.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_15.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_15.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_15_allKeys.extend(theseKeys)
                if len(_key_resp_15_allKeys):
                    key_resp_15.keys = _key_resp_15_allKeys[-1].name  # just the last key pressed
                    key_resp_15.rt = _key_resp_15_allKeys[-1].rt
                    key_resp_15.duration = _key_resp_15_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_23* updates
            
            # if text_23 is starting this frame...
            if text_23.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_23.frameNStart = frameN  # exact frame index
                text_23.tStart = t  # local t and not account for scr refresh
                text_23.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_23, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_23.started')
                # update status
                text_23.status = STARTED
                text_23.setAutoDraw(True)
            
            # if text_23 is active this frame...
            if text_23.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions7Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions7" ---
        for thisComponent in Instructions7Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions7.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_15.keys in ['', [], None]:  # No response was made
            key_resp_15.keys = None
        trials.addData('key_resp_15.keys',key_resp_15.keys)
        if key_resp_15.keys != None:  # we had a response
            trials.addData('key_resp_15.rt', key_resp_15.rt)
            trials.addData('key_resp_15.duration', key_resp_15.duration)
        # the Routine "Instructions7" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions8" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions8.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_6
        key_resp_6.keys = []
        key_resp_6.rt = []
        _key_resp_6_allKeys = []
        fill_rectangle2.setSize((rec_length, 0.05))
        # create starting attributes for key_resp_7
        key_resp_7.keys = []
        key_resp_7.rt = []
        _key_resp_7_allKeys = []
        # keep track of which components have finished
        Instructions8Components = [text_8, key_resp_6, fill_rectangle2, right_cue2, left_cue_2, dividing_line_2, key_resp_7, text_9]
        for thisComponent in Instructions8Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions8" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_8.started')
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # *key_resp_6* updates
            waitOnFlip = False
            
            # if key_resp_6 is starting this frame...
            if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_6.frameNStart = frameN  # exact frame index
                key_resp_6.tStart = t  # local t and not account for scr refresh
                key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_6.started')
                # update status
                key_resp_6.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_6.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_6.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_6_allKeys.extend(theseKeys)
                if len(_key_resp_6_allKeys):
                    key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                    key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                    key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *fill_rectangle2* updates
            
            # if fill_rectangle2 is starting this frame...
            if fill_rectangle2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fill_rectangle2.frameNStart = frameN  # exact frame index
                fill_rectangle2.tStart = t  # local t and not account for scr refresh
                fill_rectangle2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fill_rectangle2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fill_rectangle2.started')
                # update status
                fill_rectangle2.status = STARTED
                fill_rectangle2.setAutoDraw(True)
            
            # if fill_rectangle2 is active this frame...
            if fill_rectangle2.status == STARTED:
                # update params
                pass
            
            # *right_cue2* updates
            
            # if right_cue2 is starting this frame...
            if right_cue2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_cue2.frameNStart = frameN  # exact frame index
                right_cue2.tStart = t  # local t and not account for scr refresh
                right_cue2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_cue2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_cue2.started')
                # update status
                right_cue2.status = STARTED
                right_cue2.setAutoDraw(True)
            
            # if right_cue2 is active this frame...
            if right_cue2.status == STARTED:
                # update params
                pass
            
            # *left_cue_2* updates
            
            # if left_cue_2 is starting this frame...
            if left_cue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_cue_2.frameNStart = frameN  # exact frame index
                left_cue_2.tStart = t  # local t and not account for scr refresh
                left_cue_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_cue_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_cue_2.started')
                # update status
                left_cue_2.status = STARTED
                left_cue_2.setAutoDraw(True)
            
            # if left_cue_2 is active this frame...
            if left_cue_2.status == STARTED:
                # update params
                pass
            
            # *dividing_line_2* updates
            
            # if dividing_line_2 is starting this frame...
            if dividing_line_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dividing_line_2.frameNStart = frameN  # exact frame index
                dividing_line_2.tStart = t  # local t and not account for scr refresh
                dividing_line_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dividing_line_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dividing_line_2.started')
                # update status
                dividing_line_2.status = STARTED
                dividing_line_2.setAutoDraw(True)
            
            # if dividing_line_2 is active this frame...
            if dividing_line_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_7* updates
            waitOnFlip = False
            
            # if key_resp_7 is starting this frame...
            if key_resp_7.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_7.frameNStart = frameN  # exact frame index
                key_resp_7.tStart = t  # local t and not account for scr refresh
                key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_7.started')
                # update status
                key_resp_7.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_7.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_7.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_7_allKeys.extend(theseKeys)
                if len(_key_resp_7_allKeys):
                    key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                    key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                    key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_9.started')
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions8Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions8" ---
        for thisComponent in Instructions8Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions8.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_6.keys in ['', [], None]:  # No response was made
            key_resp_6.keys = None
        trials.addData('key_resp_6.keys',key_resp_6.keys)
        if key_resp_6.keys != None:  # we had a response
            trials.addData('key_resp_6.rt', key_resp_6.rt)
            trials.addData('key_resp_6.duration', key_resp_6.duration)
        # check responses
        if key_resp_7.keys in ['', [], None]:  # No response was made
            key_resp_7.keys = None
        trials.addData('key_resp_7.keys',key_resp_7.keys)
        if key_resp_7.keys != None:  # we had a response
            trials.addData('key_resp_7.rt', key_resp_7.rt)
            trials.addData('key_resp_7.duration', key_resp_7.duration)
        # the Routine "Instructions8" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions9" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions9.started', globalClock.getTime(format='float'))
        fill_rectangle_5.setSize((rec_length, 0.05))
        fill_rectangle_6.setSize((rec_length, 0.05))
        # create starting attributes for key_resp_8
        key_resp_8.keys = []
        key_resp_8.rt = []
        _key_resp_8_allKeys = []
        # keep track of which components have finished
        Instructions9Components = [text_10, fill_rectangle_5, right_cue3, left_cue3, dividing_line_3, text_11, fill_rectangle_6, right_cue3_2, left_cue3_2, dividing_line_4, key_resp_8, text_12]
        for thisComponent in Instructions9Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions9" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_10* updates
            
            # if text_10 is starting this frame...
            if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_10.frameNStart = frameN  # exact frame index
                text_10.tStart = t  # local t and not account for scr refresh
                text_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_10.started')
                # update status
                text_10.status = STARTED
                text_10.setAutoDraw(True)
            
            # if text_10 is active this frame...
            if text_10.status == STARTED:
                # update params
                pass
            
            # *fill_rectangle_5* updates
            
            # if fill_rectangle_5 is starting this frame...
            if fill_rectangle_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fill_rectangle_5.frameNStart = frameN  # exact frame index
                fill_rectangle_5.tStart = t  # local t and not account for scr refresh
                fill_rectangle_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fill_rectangle_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fill_rectangle_5.started')
                # update status
                fill_rectangle_5.status = STARTED
                fill_rectangle_5.setAutoDraw(True)
            
            # if fill_rectangle_5 is active this frame...
            if fill_rectangle_5.status == STARTED:
                # update params
                pass
            
            # *right_cue3* updates
            
            # if right_cue3 is starting this frame...
            if right_cue3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_cue3.frameNStart = frameN  # exact frame index
                right_cue3.tStart = t  # local t and not account for scr refresh
                right_cue3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_cue3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_cue3.started')
                # update status
                right_cue3.status = STARTED
                right_cue3.setAutoDraw(True)
            
            # if right_cue3 is active this frame...
            if right_cue3.status == STARTED:
                # update params
                pass
            
            # *left_cue3* updates
            
            # if left_cue3 is starting this frame...
            if left_cue3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_cue3.frameNStart = frameN  # exact frame index
                left_cue3.tStart = t  # local t and not account for scr refresh
                left_cue3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_cue3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_cue3.started')
                # update status
                left_cue3.status = STARTED
                left_cue3.setAutoDraw(True)
            
            # if left_cue3 is active this frame...
            if left_cue3.status == STARTED:
                # update params
                pass
            
            # *dividing_line_3* updates
            
            # if dividing_line_3 is starting this frame...
            if dividing_line_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dividing_line_3.frameNStart = frameN  # exact frame index
                dividing_line_3.tStart = t  # local t and not account for scr refresh
                dividing_line_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dividing_line_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dividing_line_3.started')
                # update status
                dividing_line_3.status = STARTED
                dividing_line_3.setAutoDraw(True)
            
            # if dividing_line_3 is active this frame...
            if dividing_line_3.status == STARTED:
                # update params
                pass
            
            # *text_11* updates
            
            # if text_11 is starting this frame...
            if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_11.frameNStart = frameN  # exact frame index
                text_11.tStart = t  # local t and not account for scr refresh
                text_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_11.started')
                # update status
                text_11.status = STARTED
                text_11.setAutoDraw(True)
            
            # if text_11 is active this frame...
            if text_11.status == STARTED:
                # update params
                pass
            
            # *fill_rectangle_6* updates
            
            # if fill_rectangle_6 is starting this frame...
            if fill_rectangle_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fill_rectangle_6.frameNStart = frameN  # exact frame index
                fill_rectangle_6.tStart = t  # local t and not account for scr refresh
                fill_rectangle_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fill_rectangle_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fill_rectangle_6.started')
                # update status
                fill_rectangle_6.status = STARTED
                fill_rectangle_6.setAutoDraw(True)
            
            # if fill_rectangle_6 is active this frame...
            if fill_rectangle_6.status == STARTED:
                # update params
                pass
            
            # *right_cue3_2* updates
            
            # if right_cue3_2 is starting this frame...
            if right_cue3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_cue3_2.frameNStart = frameN  # exact frame index
                right_cue3_2.tStart = t  # local t and not account for scr refresh
                right_cue3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_cue3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_cue3_2.started')
                # update status
                right_cue3_2.status = STARTED
                right_cue3_2.setAutoDraw(True)
            
            # if right_cue3_2 is active this frame...
            if right_cue3_2.status == STARTED:
                # update params
                pass
            
            # *left_cue3_2* updates
            
            # if left_cue3_2 is starting this frame...
            if left_cue3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_cue3_2.frameNStart = frameN  # exact frame index
                left_cue3_2.tStart = t  # local t and not account for scr refresh
                left_cue3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_cue3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_cue3_2.started')
                # update status
                left_cue3_2.status = STARTED
                left_cue3_2.setAutoDraw(True)
            
            # if left_cue3_2 is active this frame...
            if left_cue3_2.status == STARTED:
                # update params
                pass
            
            # *dividing_line_4* updates
            
            # if dividing_line_4 is starting this frame...
            if dividing_line_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dividing_line_4.frameNStart = frameN  # exact frame index
                dividing_line_4.tStart = t  # local t and not account for scr refresh
                dividing_line_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dividing_line_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dividing_line_4.started')
                # update status
                dividing_line_4.status = STARTED
                dividing_line_4.setAutoDraw(True)
            
            # if dividing_line_4 is active this frame...
            if dividing_line_4.status == STARTED:
                # update params
                pass
            
            # *key_resp_8* updates
            waitOnFlip = False
            
            # if key_resp_8 is starting this frame...
            if key_resp_8.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_8.frameNStart = frameN  # exact frame index
                key_resp_8.tStart = t  # local t and not account for scr refresh
                key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_8.started')
                # update status
                key_resp_8.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_8.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_8.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_8_allKeys.extend(theseKeys)
                if len(_key_resp_8_allKeys):
                    key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                    key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                    key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_12* updates
            
            # if text_12 is starting this frame...
            if text_12.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_12.frameNStart = frameN  # exact frame index
                text_12.tStart = t  # local t and not account for scr refresh
                text_12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_12.started')
                # update status
                text_12.status = STARTED
                text_12.setAutoDraw(True)
            
            # if text_12 is active this frame...
            if text_12.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions9Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions9" ---
        for thisComponent in Instructions9Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions9.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_8.keys in ['', [], None]:  # No response was made
            key_resp_8.keys = None
        trials.addData('key_resp_8.keys',key_resp_8.keys)
        if key_resp_8.keys != None:  # we had a response
            trials.addData('key_resp_8.rt', key_resp_8.rt)
            trials.addData('key_resp_8.duration', key_resp_8.duration)
        # the Routine "Instructions9" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions10" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instructions10.started', globalClock.getTime(format='float'))
        fill_rectangle_7.setSize((rec_length, 0.05))
        fill_rectangle_8.setSize((rec_length, 0.05))
        # create starting attributes for key_resp_9
        key_resp_9.keys = []
        key_resp_9.rt = []
        _key_resp_9_allKeys = []
        # keep track of which components have finished
        Instructions10Components = [text_13, fill_rectangle_7, right_cue3_3, left_cue3_3, dividing_line_6, text_14, fill_rectangle_8, right_cue3_4, left_cue3_4, dividing_line_5, key_resp_9, text_15]
        for thisComponent in Instructions10Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instructions10" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_13* updates
            
            # if text_13 is starting this frame...
            if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_13.frameNStart = frameN  # exact frame index
                text_13.tStart = t  # local t and not account for scr refresh
                text_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_13.started')
                # update status
                text_13.status = STARTED
                text_13.setAutoDraw(True)
            
            # if text_13 is active this frame...
            if text_13.status == STARTED:
                # update params
                pass
            
            # *fill_rectangle_7* updates
            
            # if fill_rectangle_7 is starting this frame...
            if fill_rectangle_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fill_rectangle_7.frameNStart = frameN  # exact frame index
                fill_rectangle_7.tStart = t  # local t and not account for scr refresh
                fill_rectangle_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fill_rectangle_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fill_rectangle_7.started')
                # update status
                fill_rectangle_7.status = STARTED
                fill_rectangle_7.setAutoDraw(True)
            
            # if fill_rectangle_7 is active this frame...
            if fill_rectangle_7.status == STARTED:
                # update params
                pass
            
            # *right_cue3_3* updates
            
            # if right_cue3_3 is starting this frame...
            if right_cue3_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_cue3_3.frameNStart = frameN  # exact frame index
                right_cue3_3.tStart = t  # local t and not account for scr refresh
                right_cue3_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_cue3_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_cue3_3.started')
                # update status
                right_cue3_3.status = STARTED
                right_cue3_3.setAutoDraw(True)
            
            # if right_cue3_3 is active this frame...
            if right_cue3_3.status == STARTED:
                # update params
                pass
            
            # *left_cue3_3* updates
            
            # if left_cue3_3 is starting this frame...
            if left_cue3_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_cue3_3.frameNStart = frameN  # exact frame index
                left_cue3_3.tStart = t  # local t and not account for scr refresh
                left_cue3_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_cue3_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_cue3_3.started')
                # update status
                left_cue3_3.status = STARTED
                left_cue3_3.setAutoDraw(True)
            
            # if left_cue3_3 is active this frame...
            if left_cue3_3.status == STARTED:
                # update params
                pass
            
            # *dividing_line_6* updates
            
            # if dividing_line_6 is starting this frame...
            if dividing_line_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dividing_line_6.frameNStart = frameN  # exact frame index
                dividing_line_6.tStart = t  # local t and not account for scr refresh
                dividing_line_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dividing_line_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dividing_line_6.started')
                # update status
                dividing_line_6.status = STARTED
                dividing_line_6.setAutoDraw(True)
            
            # if dividing_line_6 is active this frame...
            if dividing_line_6.status == STARTED:
                # update params
                pass
            
            # *text_14* updates
            
            # if text_14 is starting this frame...
            if text_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_14.frameNStart = frameN  # exact frame index
                text_14.tStart = t  # local t and not account for scr refresh
                text_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_14.started')
                # update status
                text_14.status = STARTED
                text_14.setAutoDraw(True)
            
            # if text_14 is active this frame...
            if text_14.status == STARTED:
                # update params
                pass
            
            # *fill_rectangle_8* updates
            
            # if fill_rectangle_8 is starting this frame...
            if fill_rectangle_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fill_rectangle_8.frameNStart = frameN  # exact frame index
                fill_rectangle_8.tStart = t  # local t and not account for scr refresh
                fill_rectangle_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fill_rectangle_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fill_rectangle_8.started')
                # update status
                fill_rectangle_8.status = STARTED
                fill_rectangle_8.setAutoDraw(True)
            
            # if fill_rectangle_8 is active this frame...
            if fill_rectangle_8.status == STARTED:
                # update params
                pass
            
            # *right_cue3_4* updates
            
            # if right_cue3_4 is starting this frame...
            if right_cue3_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_cue3_4.frameNStart = frameN  # exact frame index
                right_cue3_4.tStart = t  # local t and not account for scr refresh
                right_cue3_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_cue3_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_cue3_4.started')
                # update status
                right_cue3_4.status = STARTED
                right_cue3_4.setAutoDraw(True)
            
            # if right_cue3_4 is active this frame...
            if right_cue3_4.status == STARTED:
                # update params
                pass
            
            # *left_cue3_4* updates
            
            # if left_cue3_4 is starting this frame...
            if left_cue3_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_cue3_4.frameNStart = frameN  # exact frame index
                left_cue3_4.tStart = t  # local t and not account for scr refresh
                left_cue3_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_cue3_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_cue3_4.started')
                # update status
                left_cue3_4.status = STARTED
                left_cue3_4.setAutoDraw(True)
            
            # if left_cue3_4 is active this frame...
            if left_cue3_4.status == STARTED:
                # update params
                pass
            
            # *dividing_line_5* updates
            
            # if dividing_line_5 is starting this frame...
            if dividing_line_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dividing_line_5.frameNStart = frameN  # exact frame index
                dividing_line_5.tStart = t  # local t and not account for scr refresh
                dividing_line_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dividing_line_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dividing_line_5.started')
                # update status
                dividing_line_5.status = STARTED
                dividing_line_5.setAutoDraw(True)
            
            # if dividing_line_5 is active this frame...
            if dividing_line_5.status == STARTED:
                # update params
                pass
            
            # *key_resp_9* updates
            waitOnFlip = False
            
            # if key_resp_9 is starting this frame...
            if key_resp_9.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_9.frameNStart = frameN  # exact frame index
                key_resp_9.tStart = t  # local t and not account for scr refresh
                key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_9.started')
                # update status
                key_resp_9.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_9.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_9.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_9_allKeys.extend(theseKeys)
                if len(_key_resp_9_allKeys):
                    key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                    key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                    key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_15* updates
            
            # if text_15 is starting this frame...
            if text_15.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text_15.frameNStart = frameN  # exact frame index
                text_15.tStart = t  # local t and not account for scr refresh
                text_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_15.started')
                # update status
                text_15.status = STARTED
                text_15.setAutoDraw(True)
            
            # if text_15 is active this frame...
            if text_15.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instructions10Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instructions10" ---
        for thisComponent in Instructions10Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instructions10.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_9.keys in ['', [], None]:  # No response was made
            key_resp_9.keys = None
        trials.addData('key_resp_9.keys',key_resp_9.keys)
        if key_resp_9.keys != None:  # we had a response
            trials.addData('key_resp_9.rt', key_resp_9.rt)
            trials.addData('key_resp_9.duration', key_resp_9.duration)
        # the Routine "Instructions10" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_3 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('resources/taskPhases_list.csv'),
            seed=None, name='trials_3')
        thisExp.addLoop(trials_3)  # add the loop to the experiment
        thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        for thisTrial_3 in trials_3:
            currentLoop = trials_3
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
            if thisTrial_3 != None:
                for paramName in thisTrial_3:
                    globals()[paramName] = thisTrial_3[paramName]
            
            # set up handler to look after randomisation of conditions etc
            demo_prac_task = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions(taskPhases),
                seed=None, name='demo_prac_task')
            thisExp.addLoop(demo_prac_task)  # add the loop to the experiment
            thisDemo_prac_task = demo_prac_task.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisDemo_prac_task.rgb)
            if thisDemo_prac_task != None:
                for paramName in thisDemo_prac_task:
                    globals()[paramName] = thisDemo_prac_task[paramName]
            
            for thisDemo_prac_task in demo_prac_task:
                currentLoop = demo_prac_task
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisDemo_prac_task.rgb)
                if thisDemo_prac_task != None:
                    for paramName in thisDemo_prac_task:
                        globals()[paramName] = thisDemo_prac_task[paramName]
                
                # set up handler to look after randomisation of conditions etc
                repeat_practice = data.TrialHandler(nReps=5.0, method='sequential', 
                    extraInfo=expInfo, originPath=-1,
                    trialList=[None],
                    seed=None, name='repeat_practice')
                thisExp.addLoop(repeat_practice)  # add the loop to the experiment
                thisRepeat_practice = repeat_practice.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisRepeat_practice.rgb)
                if thisRepeat_practice != None:
                    for paramName in thisRepeat_practice:
                        globals()[paramName] = thisRepeat_practice[paramName]
                
                for thisRepeat_practice in repeat_practice:
                    currentLoop = repeat_practice
                    thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                    )
                    # abbreviate parameter names if possible (e.g. rgb = thisRepeat_practice.rgb)
                    if thisRepeat_practice != None:
                        for paramName in thisRepeat_practice:
                            globals()[paramName] = thisRepeat_practice[paramName]
                    
                    # --- Prepare to start Routine "intro_txt" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('intro_txt.started', globalClock.getTime(format='float'))
                    text_19.setText(introTXT)
                    # create starting attributes for key_resp_11
                    key_resp_11.keys = []
                    key_resp_11.rt = []
                    _key_resp_11_allKeys = []
                    # keep track of which components have finished
                    intro_txtComponents = [text_19, key_resp_11]
                    for thisComponent in intro_txtComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "intro_txt" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_19* updates
                        
                        # if text_19 is starting this frame...
                        if text_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_19.frameNStart = frameN  # exact frame index
                            text_19.tStart = t  # local t and not account for scr refresh
                            text_19.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_19, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_19.started')
                            # update status
                            text_19.status = STARTED
                            text_19.setAutoDraw(True)
                        
                        # if text_19 is active this frame...
                        if text_19.status == STARTED:
                            # update params
                            pass
                        
                        # *key_resp_11* updates
                        waitOnFlip = False
                        
                        # if key_resp_11 is starting this frame...
                        if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            key_resp_11.frameNStart = frameN  # exact frame index
                            key_resp_11.tStart = t  # local t and not account for scr refresh
                            key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_resp_11.started')
                            # update status
                            key_resp_11.status = STARTED
                            # keyboard checking is just starting
                            waitOnFlip = True
                            win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
                            win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
                        if key_resp_11.status == STARTED and not waitOnFlip:
                            theseKeys = key_resp_11.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                            _key_resp_11_allKeys.extend(theseKeys)
                            if len(_key_resp_11_allKeys):
                                key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                                key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                                key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                                # a response ends the routine
                                continueRoutine = False
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in intro_txtComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "intro_txt" ---
                    for thisComponent in intro_txtComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('intro_txt.stopped', globalClock.getTime(format='float'))
                    # check responses
                    if key_resp_11.keys in ['', [], None]:  # No response was made
                        key_resp_11.keys = None
                    repeat_practice.addData('key_resp_11.keys',key_resp_11.keys)
                    if key_resp_11.keys != None:  # we had a response
                        repeat_practice.addData('key_resp_11.rt', key_resp_11.rt)
                        repeat_practice.addData('key_resp_11.duration', key_resp_11.duration)
                    # the Routine "intro_txt" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    
                    # --- Prepare to start Routine "initial_wait" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('initial_wait.started', globalClock.getTime(format='float'))
                    # keep track of which components have finished
                    initial_waitComponents = [fixation_cross_4]
                    for thisComponent in initial_waitComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "initial_wait" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 2.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *fixation_cross_4* updates
                        
                        # if fixation_cross_4 is starting this frame...
                        if fixation_cross_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            fixation_cross_4.frameNStart = frameN  # exact frame index
                            fixation_cross_4.tStart = t  # local t and not account for scr refresh
                            fixation_cross_4.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(fixation_cross_4, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'fixation_cross_4.started')
                            # update status
                            fixation_cross_4.status = STARTED
                            fixation_cross_4.setAutoDraw(True)
                        
                        # if fixation_cross_4 is active this frame...
                        if fixation_cross_4.status == STARTED:
                            # update params
                            pass
                        
                        # if fixation_cross_4 is stopping this frame...
                        if fixation_cross_4.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > fixation_cross_4.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                fixation_cross_4.tStop = t  # not accounting for scr refresh
                                fixation_cross_4.tStopRefresh = tThisFlipGlobal  # on global time
                                fixation_cross_4.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_cross_4.stopped')
                                # update status
                                fixation_cross_4.status = FINISHED
                                fixation_cross_4.setAutoDraw(False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in initial_waitComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "initial_wait" ---
                    for thisComponent in initial_waitComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('initial_wait.stopped', globalClock.getTime(format='float'))
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-2.000000)
                    
                    # set up handler to look after randomisation of conditions etc
                    loop_trials = data.TrialHandler(nReps=1.0, method='random', 
                        extraInfo=expInfo, originPath=-1,
                        trialList=data.importConditions(condsFile),
                        seed=None, name='loop_trials')
                    thisExp.addLoop(loop_trials)  # add the loop to the experiment
                    thisLoop_trial = loop_trials.trialList[0]  # so we can initialise stimuli with some values
                    # abbreviate parameter names if possible (e.g. rgb = thisLoop_trial.rgb)
                    if thisLoop_trial != None:
                        for paramName in thisLoop_trial:
                            globals()[paramName] = thisLoop_trial[paramName]
                    
                    for thisLoop_trial in loop_trials:
                        currentLoop = loop_trials
                        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                        )
                        # abbreviate parameter names if possible (e.g. rgb = thisLoop_trial.rgb)
                        if thisLoop_trial != None:
                            for paramName in thisLoop_trial:
                                globals()[paramName] = thisLoop_trial[paramName]
                        
                        # --- Prepare to start Routine "memoranda" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('memoranda.started', globalClock.getTime(format='float'))
                        left_stim.setImage(left_img)
                        right_stim.setImage(right_img)
                        # keep track of which components have finished
                        memorandaComponents = [left_stim, right_stim, fixation_cross_5]
                        for thisComponent in memorandaComponents:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "memoranda" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *left_stim* updates
                            
                            # if left_stim is starting this frame...
                            if left_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                left_stim.frameNStart = frameN  # exact frame index
                                left_stim.tStart = t  # local t and not account for scr refresh
                                left_stim.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(left_stim, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'left_stim.started')
                                # update status
                                left_stim.status = STARTED
                                left_stim.setAutoDraw(True)
                            
                            # if left_stim is active this frame...
                            if left_stim.status == STARTED:
                                # update params
                                pass
                            
                            # if left_stim is stopping this frame...
                            if left_stim.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > left_stim.tStartRefresh + cue_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    left_stim.tStop = t  # not accounting for scr refresh
                                    left_stim.tStopRefresh = tThisFlipGlobal  # on global time
                                    left_stim.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'left_stim.stopped')
                                    # update status
                                    left_stim.status = FINISHED
                                    left_stim.setAutoDraw(False)
                            
                            # *right_stim* updates
                            
                            # if right_stim is starting this frame...
                            if right_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                right_stim.frameNStart = frameN  # exact frame index
                                right_stim.tStart = t  # local t and not account for scr refresh
                                right_stim.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(right_stim, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'right_stim.started')
                                # update status
                                right_stim.status = STARTED
                                right_stim.setAutoDraw(True)
                            
                            # if right_stim is active this frame...
                            if right_stim.status == STARTED:
                                # update params
                                pass
                            
                            # if right_stim is stopping this frame...
                            if right_stim.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > right_stim.tStartRefresh + cue_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    right_stim.tStop = t  # not accounting for scr refresh
                                    right_stim.tStopRefresh = tThisFlipGlobal  # on global time
                                    right_stim.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'right_stim.stopped')
                                    # update status
                                    right_stim.status = FINISHED
                                    right_stim.setAutoDraw(False)
                            
                            # *fixation_cross_5* updates
                            
                            # if fixation_cross_5 is starting this frame...
                            if fixation_cross_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                fixation_cross_5.frameNStart = frameN  # exact frame index
                                fixation_cross_5.tStart = t  # local t and not account for scr refresh
                                fixation_cross_5.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fixation_cross_5, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_cross_5.started')
                                # update status
                                fixation_cross_5.status = STARTED
                                fixation_cross_5.setAutoDraw(True)
                            
                            # if fixation_cross_5 is active this frame...
                            if fixation_cross_5.status == STARTED:
                                # update params
                                pass
                            
                            # if fixation_cross_5 is stopping this frame...
                            if fixation_cross_5.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fixation_cross_5.tStartRefresh + cue_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fixation_cross_5.tStop = t  # not accounting for scr refresh
                                    fixation_cross_5.tStopRefresh = tThisFlipGlobal  # on global time
                                    fixation_cross_5.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fixation_cross_5.stopped')
                                    # update status
                                    fixation_cross_5.status = FINISHED
                                    fixation_cross_5.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in memorandaComponents:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "memoranda" ---
                        for thisComponent in memorandaComponents:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('memoranda.stopped', globalClock.getTime(format='float'))
                        # the Routine "memoranda" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "isi1" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('isi1.started', globalClock.getTime(format='float'))
                        # keep track of which components have finished
                        isi1Components = [fixation_cross]
                        for thisComponent in isi1Components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "isi1" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *fixation_cross* updates
                            
                            # if fixation_cross is starting this frame...
                            if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                fixation_cross.frameNStart = frameN  # exact frame index
                                fixation_cross.tStart = t  # local t and not account for scr refresh
                                fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_cross.started')
                                # update status
                                fixation_cross.status = STARTED
                                fixation_cross.setAutoDraw(True)
                            
                            # if fixation_cross is active this frame...
                            if fixation_cross.status == STARTED:
                                # update params
                                pass
                            
                            # if fixation_cross is stopping this frame...
                            if fixation_cross.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fixation_cross.tStartRefresh + delay_1_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fixation_cross.tStop = t  # not accounting for scr refresh
                                    fixation_cross.tStopRefresh = tThisFlipGlobal  # on global time
                                    fixation_cross.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fixation_cross.stopped')
                                    # update status
                                    fixation_cross.status = FINISHED
                                    fixation_cross.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in isi1Components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "isi1" ---
                        for thisComponent in isi1Components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('isi1.stopped', globalClock.getTime(format='float'))
                        # the Routine "isi1" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "reliability_cue" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('reliability_cue.started', globalClock.getTime(format='float'))
                        fill_rectangle.setSize((rec_length, 0.05))
                        right_cue.setPos((right_cue_offset, 0))
                        right_cue.setSize((right_cue_length, 0.049))
                        left_cue.setPos((left_cue_offset, 0))
                        left_cue.setSize((left_cue_length, 0.049))
                        # keep track of which components have finished
                        reliability_cueComponents = [fill_rectangle, right_cue, left_cue, dividing_line]
                        for thisComponent in reliability_cueComponents:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "reliability_cue" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *fill_rectangle* updates
                            
                            # if fill_rectangle is starting this frame...
                            if fill_rectangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                fill_rectangle.frameNStart = frameN  # exact frame index
                                fill_rectangle.tStart = t  # local t and not account for scr refresh
                                fill_rectangle.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fill_rectangle, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fill_rectangle.started')
                                # update status
                                fill_rectangle.status = STARTED
                                fill_rectangle.setAutoDraw(True)
                            
                            # if fill_rectangle is active this frame...
                            if fill_rectangle.status == STARTED:
                                # update params
                                pass
                            
                            # if fill_rectangle is stopping this frame...
                            if fill_rectangle.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fill_rectangle.tStartRefresh + reliability_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fill_rectangle.tStop = t  # not accounting for scr refresh
                                    fill_rectangle.tStopRefresh = tThisFlipGlobal  # on global time
                                    fill_rectangle.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fill_rectangle.stopped')
                                    # update status
                                    fill_rectangle.status = FINISHED
                                    fill_rectangle.setAutoDraw(False)
                            
                            # *right_cue* updates
                            
                            # if right_cue is starting this frame...
                            if right_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                right_cue.frameNStart = frameN  # exact frame index
                                right_cue.tStart = t  # local t and not account for scr refresh
                                right_cue.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(right_cue, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'right_cue.started')
                                # update status
                                right_cue.status = STARTED
                                right_cue.setAutoDraw(True)
                            
                            # if right_cue is active this frame...
                            if right_cue.status == STARTED:
                                # update params
                                pass
                            
                            # if right_cue is stopping this frame...
                            if right_cue.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > right_cue.tStartRefresh + reliability_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    right_cue.tStop = t  # not accounting for scr refresh
                                    right_cue.tStopRefresh = tThisFlipGlobal  # on global time
                                    right_cue.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'right_cue.stopped')
                                    # update status
                                    right_cue.status = FINISHED
                                    right_cue.setAutoDraw(False)
                            
                            # *left_cue* updates
                            
                            # if left_cue is starting this frame...
                            if left_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                left_cue.frameNStart = frameN  # exact frame index
                                left_cue.tStart = t  # local t and not account for scr refresh
                                left_cue.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(left_cue, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'left_cue.started')
                                # update status
                                left_cue.status = STARTED
                                left_cue.setAutoDraw(True)
                            
                            # if left_cue is active this frame...
                            if left_cue.status == STARTED:
                                # update params
                                pass
                            
                            # if left_cue is stopping this frame...
                            if left_cue.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > left_cue.tStartRefresh + reliability_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    left_cue.tStop = t  # not accounting for scr refresh
                                    left_cue.tStopRefresh = tThisFlipGlobal  # on global time
                                    left_cue.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'left_cue.stopped')
                                    # update status
                                    left_cue.status = FINISHED
                                    left_cue.setAutoDraw(False)
                            
                            # *dividing_line* updates
                            
                            # if dividing_line is starting this frame...
                            if dividing_line.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                dividing_line.frameNStart = frameN  # exact frame index
                                dividing_line.tStart = t  # local t and not account for scr refresh
                                dividing_line.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(dividing_line, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'dividing_line.started')
                                # update status
                                dividing_line.status = STARTED
                                dividing_line.setAutoDraw(True)
                            
                            # if dividing_line is active this frame...
                            if dividing_line.status == STARTED:
                                # update params
                                pass
                            
                            # if dividing_line is stopping this frame...
                            if dividing_line.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > dividing_line.tStartRefresh + reliability_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    dividing_line.tStop = t  # not accounting for scr refresh
                                    dividing_line.tStopRefresh = tThisFlipGlobal  # on global time
                                    dividing_line.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'dividing_line.stopped')
                                    # update status
                                    dividing_line.status = FINISHED
                                    dividing_line.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in reliability_cueComponents:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "reliability_cue" ---
                        for thisComponent in reliability_cueComponents:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('reliability_cue.stopped', globalClock.getTime(format='float'))
                        # the Routine "reliability_cue" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "isi2" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('isi2.started', globalClock.getTime(format='float'))
                        # keep track of which components have finished
                        isi2Components = [fixation_cross2]
                        for thisComponent in isi2Components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "isi2" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *fixation_cross2* updates
                            
                            # if fixation_cross2 is starting this frame...
                            if fixation_cross2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                fixation_cross2.frameNStart = frameN  # exact frame index
                                fixation_cross2.tStart = t  # local t and not account for scr refresh
                                fixation_cross2.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fixation_cross2, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_cross2.started')
                                # update status
                                fixation_cross2.status = STARTED
                                fixation_cross2.setAutoDraw(True)
                            
                            # if fixation_cross2 is active this frame...
                            if fixation_cross2.status == STARTED:
                                # update params
                                pass
                            
                            # if fixation_cross2 is stopping this frame...
                            if fixation_cross2.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fixation_cross2.tStartRefresh + delay_2_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fixation_cross2.tStop = t  # not accounting for scr refresh
                                    fixation_cross2.tStopRefresh = tThisFlipGlobal  # on global time
                                    fixation_cross2.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fixation_cross2.stopped')
                                    # update status
                                    fixation_cross2.status = FINISHED
                                    fixation_cross2.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in isi2Components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "isi2" ---
                        for thisComponent in isi2Components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('isi2.stopped', globalClock.getTime(format='float'))
                        # the Routine "isi2" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "ping" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('ping.started', globalClock.getTime(format='float'))
                        ping_im.setImage(ping_img)
                        ping_im_2.setImage(ping_img)
                        ping_im_3.setImage(ping_img)
                        # keep track of which components have finished
                        pingComponents = [ping_im, fixation_cross2_2, ping_im_2, fixation_cross2_3, ping_im_3]
                        for thisComponent in pingComponents:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "ping" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *ping_im* updates
                            
                            # if ping_im is starting this frame...
                            if ping_im.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                ping_im.frameNStart = frameN  # exact frame index
                                ping_im.tStart = t  # local t and not account for scr refresh
                                ping_im.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(ping_im, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'ping_im.started')
                                # update status
                                ping_im.status = STARTED
                                ping_im.setAutoDraw(True)
                            
                            # if ping_im is active this frame...
                            if ping_im.status == STARTED:
                                # update params
                                pass
                            
                            # if ping_im is stopping this frame...
                            if ping_im.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > ping_im.tStartRefresh + ping_im_flash1-frameTolerance:
                                    # keep track of stop time/frame for later
                                    ping_im.tStop = t  # not accounting for scr refresh
                                    ping_im.tStopRefresh = tThisFlipGlobal  # on global time
                                    ping_im.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'ping_im.stopped')
                                    # update status
                                    ping_im.status = FINISHED
                                    ping_im.setAutoDraw(False)
                            
                            # *fixation_cross2_2* updates
                            
                            # if fixation_cross2_2 is starting this frame...
                            if fixation_cross2_2.status == NOT_STARTED and tThisFlip >= .2-frameTolerance:
                                # keep track of start time/frame for later
                                fixation_cross2_2.frameNStart = frameN  # exact frame index
                                fixation_cross2_2.tStart = t  # local t and not account for scr refresh
                                fixation_cross2_2.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fixation_cross2_2, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_cross2_2.started')
                                # update status
                                fixation_cross2_2.status = STARTED
                                fixation_cross2_2.setAutoDraw(True)
                            
                            # if fixation_cross2_2 is active this frame...
                            if fixation_cross2_2.status == STARTED:
                                # update params
                                pass
                            
                            # if fixation_cross2_2 is stopping this frame...
                            if fixation_cross2_2.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fixation_cross2_2.tStartRefresh + ping_im_flash2-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fixation_cross2_2.tStop = t  # not accounting for scr refresh
                                    fixation_cross2_2.tStopRefresh = tThisFlipGlobal  # on global time
                                    fixation_cross2_2.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fixation_cross2_2.stopped')
                                    # update status
                                    fixation_cross2_2.status = FINISHED
                                    fixation_cross2_2.setAutoDraw(False)
                            
                            # *ping_im_2* updates
                            
                            # if ping_im_2 is starting this frame...
                            if ping_im_2.status == NOT_STARTED and tThisFlip >= .4-frameTolerance:
                                # keep track of start time/frame for later
                                ping_im_2.frameNStart = frameN  # exact frame index
                                ping_im_2.tStart = t  # local t and not account for scr refresh
                                ping_im_2.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(ping_im_2, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'ping_im_2.started')
                                # update status
                                ping_im_2.status = STARTED
                                ping_im_2.setAutoDraw(True)
                            
                            # if ping_im_2 is active this frame...
                            if ping_im_2.status == STARTED:
                                # update params
                                pass
                            
                            # if ping_im_2 is stopping this frame...
                            if ping_im_2.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > ping_im_2.tStartRefresh + ping_im_flash3-frameTolerance:
                                    # keep track of stop time/frame for later
                                    ping_im_2.tStop = t  # not accounting for scr refresh
                                    ping_im_2.tStopRefresh = tThisFlipGlobal  # on global time
                                    ping_im_2.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'ping_im_2.stopped')
                                    # update status
                                    ping_im_2.status = FINISHED
                                    ping_im_2.setAutoDraw(False)
                            
                            # *fixation_cross2_3* updates
                            
                            # if fixation_cross2_3 is starting this frame...
                            if fixation_cross2_3.status == NOT_STARTED and tThisFlip >= .6-frameTolerance:
                                # keep track of start time/frame for later
                                fixation_cross2_3.frameNStart = frameN  # exact frame index
                                fixation_cross2_3.tStart = t  # local t and not account for scr refresh
                                fixation_cross2_3.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fixation_cross2_3, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_cross2_3.started')
                                # update status
                                fixation_cross2_3.status = STARTED
                                fixation_cross2_3.setAutoDraw(True)
                            
                            # if fixation_cross2_3 is active this frame...
                            if fixation_cross2_3.status == STARTED:
                                # update params
                                pass
                            
                            # if fixation_cross2_3 is stopping this frame...
                            if fixation_cross2_3.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fixation_cross2_3.tStartRefresh + ping_im_flash4-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fixation_cross2_3.tStop = t  # not accounting for scr refresh
                                    fixation_cross2_3.tStopRefresh = tThisFlipGlobal  # on global time
                                    fixation_cross2_3.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fixation_cross2_3.stopped')
                                    # update status
                                    fixation_cross2_3.status = FINISHED
                                    fixation_cross2_3.setAutoDraw(False)
                            
                            # *ping_im_3* updates
                            
                            # if ping_im_3 is starting this frame...
                            if ping_im_3.status == NOT_STARTED and tThisFlip >= .8-frameTolerance:
                                # keep track of start time/frame for later
                                ping_im_3.frameNStart = frameN  # exact frame index
                                ping_im_3.tStart = t  # local t and not account for scr refresh
                                ping_im_3.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(ping_im_3, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'ping_im_3.started')
                                # update status
                                ping_im_3.status = STARTED
                                ping_im_3.setAutoDraw(True)
                            
                            # if ping_im_3 is active this frame...
                            if ping_im_3.status == STARTED:
                                # update params
                                pass
                            
                            # if ping_im_3 is stopping this frame...
                            if ping_im_3.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > ping_im_3.tStartRefresh + ping_im_flash5-frameTolerance:
                                    # keep track of stop time/frame for later
                                    ping_im_3.tStop = t  # not accounting for scr refresh
                                    ping_im_3.tStopRefresh = tThisFlipGlobal  # on global time
                                    ping_im_3.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'ping_im_3.stopped')
                                    # update status
                                    ping_im_3.status = FINISHED
                                    ping_im_3.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in pingComponents:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "ping" ---
                        for thisComponent in pingComponents:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('ping.stopped', globalClock.getTime(format='float'))
                        # the Routine "ping" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "noise_mask" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('noise_mask.started', globalClock.getTime(format='float'))
                        # keep track of which components have finished
                        noise_maskComponents = [noise_mask_im]
                        for thisComponent in noise_maskComponents:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "noise_mask" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *noise_mask_im* updates
                            
                            # if noise_mask_im is starting this frame...
                            if noise_mask_im.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                noise_mask_im.frameNStart = frameN  # exact frame index
                                noise_mask_im.tStart = t  # local t and not account for scr refresh
                                noise_mask_im.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(noise_mask_im, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'noise_mask_im.started')
                                # update status
                                noise_mask_im.status = STARTED
                                noise_mask_im.setAutoDraw(True)
                            
                            # if noise_mask_im is active this frame...
                            if noise_mask_im.status == STARTED:
                                # update params
                                pass
                            
                            # if noise_mask_im is stopping this frame...
                            if noise_mask_im.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > noise_mask_im.tStartRefresh + noise_mask_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    noise_mask_im.tStop = t  # not accounting for scr refresh
                                    noise_mask_im.tStopRefresh = tThisFlipGlobal  # on global time
                                    noise_mask_im.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'noise_mask_im.stopped')
                                    # update status
                                    noise_mask_im.status = FINISHED
                                    noise_mask_im.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in noise_maskComponents:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "noise_mask" ---
                        for thisComponent in noise_maskComponents:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('noise_mask.stopped', globalClock.getTime(format='float'))
                        # the Routine "noise_mask" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "isi3" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('isi3.started', globalClock.getTime(format='float'))
                        # keep track of which components have finished
                        isi3Components = [fixation_cross_3]
                        for thisComponent in isi3Components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "isi3" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *fixation_cross_3* updates
                            
                            # if fixation_cross_3 is starting this frame...
                            if fixation_cross_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                fixation_cross_3.frameNStart = frameN  # exact frame index
                                fixation_cross_3.tStart = t  # local t and not account for scr refresh
                                fixation_cross_3.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fixation_cross_3, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_cross_3.started')
                                # update status
                                fixation_cross_3.status = STARTED
                                fixation_cross_3.setAutoDraw(True)
                            
                            # if fixation_cross_3 is active this frame...
                            if fixation_cross_3.status == STARTED:
                                # update params
                                pass
                            
                            # if fixation_cross_3 is stopping this frame...
                            if fixation_cross_3.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fixation_cross_3.tStartRefresh + delay_3_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fixation_cross_3.tStop = t  # not accounting for scr refresh
                                    fixation_cross_3.tStopRefresh = tThisFlipGlobal  # on global time
                                    fixation_cross_3.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fixation_cross_3.stopped')
                                    # update status
                                    fixation_cross_3.status = FINISHED
                                    fixation_cross_3.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in isi3Components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "isi3" ---
                        for thisComponent in isi3Components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('isi3.stopped', globalClock.getTime(format='float'))
                        # the Routine "isi3" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "probe" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('probe.started', globalClock.getTime(format='float'))
                        distractor_mem_stim.setPos((dismem_x, dismem_y))
                        distractor_mem_stim.setImage(distractor_stim)
                        memoranda_stim.setPos((mem_x, mem_y))
                        memoranda_stim.setImage(mem_stim)
                        distractor_im_0.setPos((dis0_x, dis0_y))
                        distractor_im_0.setImage(distractor_0)
                        distractor_im_1.setPos((dis1_x, dis1_y))
                        distractor_im_1.setImage(distractor_1)
                        distractor_im_2.setPos((dis2_x, dis2_y))
                        distractor_im_2.setImage(distractor_2)
                        distractor_im_3.setPos((dis3_x, dis3_y))
                        distractor_im_3.setImage(distractor_3)
                        # setup some python lists for storing info about the mouse
                        mouse.x = []
                        mouse.y = []
                        mouse.leftButton = []
                        mouse.midButton = []
                        mouse.rightButton = []
                        mouse.time = []
                        mouse.clicked_name = []
                        gotValidClick = False  # until a click is received
                        mem_cue.setImage(cue_img)
                        text_24.setText('instruct_guide_txt')
                        # Run 'Begin Routine' code from code_2
                        
                        // inserted the crib sheet stuff
                        thisExp=psychoJS.experiment;
                        win=psychoJS.window;
                        event=psychoJS.eventManager;
                        
                        
                        //hide mouse
                        document.body.style.cursor='none'
                        
                        //exit if on mobile
                        if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
                          quitPsychoJS('Mobile device detected. Goodbye!', false)
                        }
                        # keep track of which components have finished
                        probeComponents = [picker, distractor_mem_stim, memoranda_stim, distractor_im_0, distractor_im_1, distractor_im_2, distractor_im_3, mouse, mem_cue, text_24]
                        for thisComponent in probeComponents:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "probe" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *picker* updates
                            
                            # if picker is starting this frame...
                            if picker.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                picker.frameNStart = frameN  # exact frame index
                                picker.tStart = t  # local t and not account for scr refresh
                                picker.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(picker, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'picker.started')
                                # update status
                                picker.status = STARTED
                                picker.setAutoDraw(True)
                            
                            # if picker is active this frame...
                            if picker.status == STARTED:
                                # update params
                                pass
                            
                            # if picker is stopping this frame...
                            if picker.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > picker.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    picker.tStop = t  # not accounting for scr refresh
                                    picker.tStopRefresh = tThisFlipGlobal  # on global time
                                    picker.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'picker.stopped')
                                    # update status
                                    picker.status = FINISHED
                                    picker.setAutoDraw(False)
                            
                            # *distractor_mem_stim* updates
                            
                            # if distractor_mem_stim is starting this frame...
                            if distractor_mem_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                distractor_mem_stim.frameNStart = frameN  # exact frame index
                                distractor_mem_stim.tStart = t  # local t and not account for scr refresh
                                distractor_mem_stim.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(distractor_mem_stim, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'distractor_mem_stim.started')
                                # update status
                                distractor_mem_stim.status = STARTED
                                distractor_mem_stim.setAutoDraw(True)
                            
                            # if distractor_mem_stim is active this frame...
                            if distractor_mem_stim.status == STARTED:
                                # update params
                                pass
                            
                            # if distractor_mem_stim is stopping this frame...
                            if distractor_mem_stim.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > distractor_mem_stim.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    distractor_mem_stim.tStop = t  # not accounting for scr refresh
                                    distractor_mem_stim.tStopRefresh = tThisFlipGlobal  # on global time
                                    distractor_mem_stim.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'distractor_mem_stim.stopped')
                                    # update status
                                    distractor_mem_stim.status = FINISHED
                                    distractor_mem_stim.setAutoDraw(False)
                            
                            # *memoranda_stim* updates
                            
                            # if memoranda_stim is starting this frame...
                            if memoranda_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                memoranda_stim.frameNStart = frameN  # exact frame index
                                memoranda_stim.tStart = t  # local t and not account for scr refresh
                                memoranda_stim.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(memoranda_stim, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'memoranda_stim.started')
                                # update status
                                memoranda_stim.status = STARTED
                                memoranda_stim.setAutoDraw(True)
                            
                            # if memoranda_stim is active this frame...
                            if memoranda_stim.status == STARTED:
                                # update params
                                pass
                            
                            # if memoranda_stim is stopping this frame...
                            if memoranda_stim.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > memoranda_stim.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    memoranda_stim.tStop = t  # not accounting for scr refresh
                                    memoranda_stim.tStopRefresh = tThisFlipGlobal  # on global time
                                    memoranda_stim.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'memoranda_stim.stopped')
                                    # update status
                                    memoranda_stim.status = FINISHED
                                    memoranda_stim.setAutoDraw(False)
                            
                            # *distractor_im_0* updates
                            
                            # if distractor_im_0 is starting this frame...
                            if distractor_im_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                distractor_im_0.frameNStart = frameN  # exact frame index
                                distractor_im_0.tStart = t  # local t and not account for scr refresh
                                distractor_im_0.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(distractor_im_0, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'distractor_im_0.started')
                                # update status
                                distractor_im_0.status = STARTED
                                distractor_im_0.setAutoDraw(True)
                            
                            # if distractor_im_0 is active this frame...
                            if distractor_im_0.status == STARTED:
                                # update params
                                pass
                            
                            # if distractor_im_0 is stopping this frame...
                            if distractor_im_0.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > distractor_im_0.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    distractor_im_0.tStop = t  # not accounting for scr refresh
                                    distractor_im_0.tStopRefresh = tThisFlipGlobal  # on global time
                                    distractor_im_0.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'distractor_im_0.stopped')
                                    # update status
                                    distractor_im_0.status = FINISHED
                                    distractor_im_0.setAutoDraw(False)
                            
                            # *distractor_im_1* updates
                            
                            # if distractor_im_1 is starting this frame...
                            if distractor_im_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                distractor_im_1.frameNStart = frameN  # exact frame index
                                distractor_im_1.tStart = t  # local t and not account for scr refresh
                                distractor_im_1.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(distractor_im_1, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'distractor_im_1.started')
                                # update status
                                distractor_im_1.status = STARTED
                                distractor_im_1.setAutoDraw(True)
                            
                            # if distractor_im_1 is active this frame...
                            if distractor_im_1.status == STARTED:
                                # update params
                                pass
                            
                            # if distractor_im_1 is stopping this frame...
                            if distractor_im_1.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > distractor_im_1.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    distractor_im_1.tStop = t  # not accounting for scr refresh
                                    distractor_im_1.tStopRefresh = tThisFlipGlobal  # on global time
                                    distractor_im_1.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'distractor_im_1.stopped')
                                    # update status
                                    distractor_im_1.status = FINISHED
                                    distractor_im_1.setAutoDraw(False)
                            
                            # *distractor_im_2* updates
                            
                            # if distractor_im_2 is starting this frame...
                            if distractor_im_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                distractor_im_2.frameNStart = frameN  # exact frame index
                                distractor_im_2.tStart = t  # local t and not account for scr refresh
                                distractor_im_2.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(distractor_im_2, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'distractor_im_2.started')
                                # update status
                                distractor_im_2.status = STARTED
                                distractor_im_2.setAutoDraw(True)
                            
                            # if distractor_im_2 is active this frame...
                            if distractor_im_2.status == STARTED:
                                # update params
                                pass
                            
                            # if distractor_im_2 is stopping this frame...
                            if distractor_im_2.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > distractor_im_2.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    distractor_im_2.tStop = t  # not accounting for scr refresh
                                    distractor_im_2.tStopRefresh = tThisFlipGlobal  # on global time
                                    distractor_im_2.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'distractor_im_2.stopped')
                                    # update status
                                    distractor_im_2.status = FINISHED
                                    distractor_im_2.setAutoDraw(False)
                            
                            # *distractor_im_3* updates
                            
                            # if distractor_im_3 is starting this frame...
                            if distractor_im_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                distractor_im_3.frameNStart = frameN  # exact frame index
                                distractor_im_3.tStart = t  # local t and not account for scr refresh
                                distractor_im_3.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(distractor_im_3, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'distractor_im_3.started')
                                # update status
                                distractor_im_3.status = STARTED
                                distractor_im_3.setAutoDraw(True)
                            
                            # if distractor_im_3 is active this frame...
                            if distractor_im_3.status == STARTED:
                                # update params
                                pass
                            
                            # if distractor_im_3 is stopping this frame...
                            if distractor_im_3.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > distractor_im_3.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    distractor_im_3.tStop = t  # not accounting for scr refresh
                                    distractor_im_3.tStopRefresh = tThisFlipGlobal  # on global time
                                    distractor_im_3.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'distractor_im_3.stopped')
                                    # update status
                                    distractor_im_3.status = FINISHED
                                    distractor_im_3.setAutoDraw(False)
                            # *mouse* updates
                            
                            # if mouse is starting this frame...
                            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                mouse.frameNStart = frameN  # exact frame index
                                mouse.tStart = t  # local t and not account for scr refresh
                                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.addData('mouse.started', t)
                                # update status
                                mouse.status = STARTED
                                mouse.mouseClock.reset()
                                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
                            
                            # if mouse is stopping this frame...
                            if mouse.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > mouse.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    mouse.tStop = t  # not accounting for scr refresh
                                    mouse.tStopRefresh = tThisFlipGlobal  # on global time
                                    mouse.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.addData('mouse.stopped', t)
                                    # update status
                                    mouse.status = FINISHED
                            if mouse.status == STARTED:  # only update if started and not finished!
                                buttons = mouse.getPressed()
                                if buttons != prevButtonState:  # button state changed?
                                    prevButtonState = buttons
                                    if sum(buttons) > 0:  # state changed to a new click
                                        # check if the mouse was inside our 'clickable' objects
                                        gotValidClick = False
                                        clickableList = environmenttools.getFromNames([memoranda_stim, distractor_mem_stim, distractor_im_0, distractor_im_1, distractor_im_2, distractor_im_3], namespace=locals())
                                        for obj in clickableList:
                                            # is this object clicked on?
                                            if obj.contains(mouse):
                                                gotValidClick = True
                                                mouse.clicked_name.append(obj.name)
                                        x, y = mouse.getPos()
                                        mouse.x.append(x)
                                        mouse.y.append(y)
                                        buttons = mouse.getPressed()
                                        mouse.leftButton.append(buttons[0])
                                        mouse.midButton.append(buttons[1])
                                        mouse.rightButton.append(buttons[2])
                                        mouse.time.append(mouse.mouseClock.getTime())
                            
                            # *mem_cue* updates
                            
                            # if mem_cue is starting this frame...
                            if mem_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                mem_cue.frameNStart = frameN  # exact frame index
                                mem_cue.tStart = t  # local t and not account for scr refresh
                                mem_cue.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(mem_cue, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'mem_cue.started')
                                # update status
                                mem_cue.status = STARTED
                                mem_cue.setAutoDraw(True)
                            
                            # if mem_cue is active this frame...
                            if mem_cue.status == STARTED:
                                # update params
                                pass
                            
                            # if mem_cue is stopping this frame...
                            if mem_cue.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > mem_cue.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    mem_cue.tStop = t  # not accounting for scr refresh
                                    mem_cue.tStopRefresh = tThisFlipGlobal  # on global time
                                    mem_cue.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'mem_cue.stopped')
                                    # update status
                                    mem_cue.status = FINISHED
                                    mem_cue.setAutoDraw(False)
                            
                            # *text_24* updates
                            
                            # if text_24 is starting this frame...
                            if text_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                text_24.frameNStart = frameN  # exact frame index
                                text_24.tStart = t  # local t and not account for scr refresh
                                text_24.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(text_24, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'text_24.started')
                                # update status
                                text_24.status = STARTED
                                text_24.setAutoDraw(True)
                            
                            # if text_24 is active this frame...
                            if text_24.status == STARTED:
                                # update params
                                pass
                            
                            # if text_24 is stopping this frame...
                            if text_24.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > text_24.tStartRefresh + probe_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    text_24.tStop = t  # not accounting for scr refresh
                                    text_24.tStopRefresh = tThisFlipGlobal  # on global time
                                    text_24.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'text_24.stopped')
                                    # update status
                                    text_24.status = FINISHED
                                    text_24.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in probeComponents:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "probe" ---
                        for thisComponent in probeComponents:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('probe.stopped', globalClock.getTime(format='float'))
                        # store data for loop_trials (TrialHandler)
                        loop_trials.addData('mouse.x', mouse.x)
                        loop_trials.addData('mouse.y', mouse.y)
                        loop_trials.addData('mouse.leftButton', mouse.leftButton)
                        loop_trials.addData('mouse.midButton', mouse.midButton)
                        loop_trials.addData('mouse.rightButton', mouse.rightButton)
                        loop_trials.addData('mouse.time', mouse.time)
                        loop_trials.addData('mouse.clicked_name', mouse.clicked_name)
                        # the Routine "probe" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "block_break" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('block_break.started', globalClock.getTime(format='float'))
                        break_txt.setText('take_break_txt')
                        # create starting attributes for key_resp
                        key_resp.keys = []
                        key_resp.rt = []
                        _key_resp_allKeys = []
                        # keep track of which components have finished
                        block_breakComponents = [break_txt, key_resp]
                        for thisComponent in block_breakComponents:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "block_break" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *break_txt* updates
                            
                            # if break_txt is starting this frame...
                            if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                break_txt.frameNStart = frameN  # exact frame index
                                break_txt.tStart = t  # local t and not account for scr refresh
                                break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'break_txt.started')
                                # update status
                                break_txt.status = STARTED
                                break_txt.setAutoDraw(True)
                            
                            # if break_txt is active this frame...
                            if break_txt.status == STARTED:
                                # update params
                                pass
                            
                            # *key_resp* updates
                            waitOnFlip = False
                            
                            # if key_resp is starting this frame...
                            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                key_resp.frameNStart = frameN  # exact frame index
                                key_resp.tStart = t  # local t and not account for scr refresh
                                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'key_resp.started')
                                # update status
                                key_resp.status = STARTED
                                # keyboard checking is just starting
                                waitOnFlip = True
                                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                            if key_resp.status == STARTED and not waitOnFlip:
                                theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                                _key_resp_allKeys.extend(theseKeys)
                                if len(_key_resp_allKeys):
                                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                                    key_resp.rt = _key_resp_allKeys[-1].rt
                                    key_resp.duration = _key_resp_allKeys[-1].duration
                                    # a response ends the routine
                                    continueRoutine = False
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in block_breakComponents:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "block_break" ---
                        for thisComponent in block_breakComponents:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('block_break.stopped', globalClock.getTime(format='float'))
                        # check responses
                        if key_resp.keys in ['', [], None]:  # No response was made
                            key_resp.keys = None
                        loop_trials.addData('key_resp.keys',key_resp.keys)
                        if key_resp.keys != None:  # we had a response
                            loop_trials.addData('key_resp.rt', key_resp.rt)
                            loop_trials.addData('key_resp.duration', key_resp.duration)
                        # the Routine "block_break" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "ITI" ---
                        continueRoutine = True
                        # update component parameters for each repeat
                        thisExp.addData('ITI.started', globalClock.getTime(format='float'))
                        # keep track of which components have finished
                        ITIComponents = [fixation_iti]
                        for thisComponent in ITIComponents:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "ITI" ---
                        routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *fixation_iti* updates
                            
                            # if fixation_iti is starting this frame...
                            if fixation_iti.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                fixation_iti.frameNStart = frameN  # exact frame index
                                fixation_iti.tStart = t  # local t and not account for scr refresh
                                fixation_iti.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(fixation_iti, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'fixation_iti.started')
                                # update status
                                fixation_iti.status = STARTED
                                fixation_iti.setAutoDraw(True)
                            
                            # if fixation_iti is active this frame...
                            if fixation_iti.status == STARTED:
                                # update params
                                pass
                            
                            # if fixation_iti is stopping this frame...
                            if fixation_iti.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > fixation_iti.tStartRefresh + iti_dur-frameTolerance:
                                    # keep track of stop time/frame for later
                                    fixation_iti.tStop = t  # not accounting for scr refresh
                                    fixation_iti.tStopRefresh = tThisFlipGlobal  # on global time
                                    fixation_iti.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'fixation_iti.stopped')
                                    # update status
                                    fixation_iti.status = FINISHED
                                    fixation_iti.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in ITIComponents:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "ITI" ---
                        for thisComponent in ITIComponents:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        thisExp.addData('ITI.stopped', globalClock.getTime(format='float'))
                        # the Routine "ITI" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        thisExp.nextEntry()
                        
                        if thisSession is not None:
                            # if running in a Session with a Liaison client, send data up to now
                            thisSession.sendExperimentData()
                    # completed 1.0 repeats of 'loop_trials'
                    
                    
                    # --- Prepare to start Routine "repeat_practice_code" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('repeat_practice_code.started', globalClock.getTime(format='float'))
                    # create starting attributes for key_resp_12
                    key_resp_12.keys = []
                    key_resp_12.rt = []
                    _key_resp_12_allKeys = []
                    # keep track of which components have finished
                    repeat_practice_codeComponents = [repeat_prac_text, key_resp_12]
                    for thisComponent in repeat_practice_codeComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "repeat_practice_code" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *repeat_prac_text* updates
                        
                        # if repeat_prac_text is starting this frame...
                        if repeat_prac_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            repeat_prac_text.frameNStart = frameN  # exact frame index
                            repeat_prac_text.tStart = t  # local t and not account for scr refresh
                            repeat_prac_text.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(repeat_prac_text, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'repeat_prac_text.started')
                            # update status
                            repeat_prac_text.status = STARTED
                            repeat_prac_text.setAutoDraw(True)
                        
                        # if repeat_prac_text is active this frame...
                        if repeat_prac_text.status == STARTED:
                            # update params
                            pass
                        
                        # *key_resp_12* updates
                        waitOnFlip = False
                        
                        # if key_resp_12 is starting this frame...
                        if key_resp_12.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                            # keep track of start time/frame for later
                            key_resp_12.frameNStart = frameN  # exact frame index
                            key_resp_12.tStart = t  # local t and not account for scr refresh
                            key_resp_12.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(key_resp_12, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_resp_12.started')
                            # update status
                            key_resp_12.status = STARTED
                            # keyboard checking is just starting
                            waitOnFlip = True
                            win.callOnFlip(key_resp_12.clock.reset)  # t=0 on next screen flip
                            win.callOnFlip(key_resp_12.clearEvents, eventType='keyboard')  # clear events on next screen flip
                        if key_resp_12.status == STARTED and not waitOnFlip:
                            theseKeys = key_resp_12.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                            _key_resp_12_allKeys.extend(theseKeys)
                            if len(_key_resp_12_allKeys):
                                key_resp_12.keys = _key_resp_12_allKeys[-1].name  # just the last key pressed
                                key_resp_12.rt = _key_resp_12_allKeys[-1].rt
                                key_resp_12.duration = _key_resp_12_allKeys[-1].duration
                                # a response ends the routine
                                continueRoutine = False
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in repeat_practice_codeComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "repeat_practice_code" ---
                    for thisComponent in repeat_practice_codeComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('repeat_practice_code.stopped', globalClock.getTime(format='float'))
                    # check responses
                    if key_resp_12.keys in ['', [], None]:  # No response was made
                        key_resp_12.keys = None
                    repeat_practice.addData('key_resp_12.keys',key_resp_12.keys)
                    if key_resp_12.keys != None:  # we had a response
                        repeat_practice.addData('key_resp_12.rt', key_resp_12.rt)
                        repeat_practice.addData('key_resp_12.duration', key_resp_12.duration)
                    # the Routine "repeat_practice_code" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed 5.0 repeats of 'repeat_practice'
                
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 1.0 repeats of 'demo_prac_task'
            
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_3'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "Instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instructions.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_17
    key_resp_17.keys = []
    key_resp_17.rt = []
    _key_resp_17_allKeys = []
    # keep track of which components have finished
    InstructionsComponents = [Instructions_to_image_memory_test, key_resp_17]
    for thisComponent in InstructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Instructions_to_image_memory_test* updates
        
        # if Instructions_to_image_memory_test is starting this frame...
        if Instructions_to_image_memory_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Instructions_to_image_memory_test.frameNStart = frameN  # exact frame index
            Instructions_to_image_memory_test.tStart = t  # local t and not account for scr refresh
            Instructions_to_image_memory_test.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instructions_to_image_memory_test, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instructions_to_image_memory_test.started')
            # update status
            Instructions_to_image_memory_test.status = STARTED
            Instructions_to_image_memory_test.setAutoDraw(True)
        
        # if Instructions_to_image_memory_test is active this frame...
        if Instructions_to_image_memory_test.status == STARTED:
            # update params
            pass
        
        # if Instructions_to_image_memory_test is stopping this frame...
        if Instructions_to_image_memory_test.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Instructions_to_image_memory_test.tStartRefresh + 300-frameTolerance:
                # keep track of stop time/frame for later
                Instructions_to_image_memory_test.tStop = t  # not accounting for scr refresh
                Instructions_to_image_memory_test.tStopRefresh = tThisFlipGlobal  # on global time
                Instructions_to_image_memory_test.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Instructions_to_image_memory_test.stopped')
                # update status
                Instructions_to_image_memory_test.status = FINISHED
                Instructions_to_image_memory_test.setAutoDraw(False)
        
        # *key_resp_17* updates
        waitOnFlip = False
        
        # if key_resp_17 is starting this frame...
        if key_resp_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_17.frameNStart = frameN  # exact frame index
            key_resp_17.tStart = t  # local t and not account for scr refresh
            key_resp_17.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_17, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_17.started')
            # update status
            key_resp_17.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_17.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_17.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_17.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_17.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_17_allKeys.extend(theseKeys)
            if len(_key_resp_17_allKeys):
                key_resp_17.keys = _key_resp_17_allKeys[-1].name  # just the last key pressed
                key_resp_17.rt = _key_resp_17_allKeys[-1].rt
                key_resp_17.duration = _key_resp_17_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in InstructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instructions.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_17.keys in ['', [], None]:  # No response was made
        key_resp_17.keys = None
    thisExp.addData('key_resp_17.keys',key_resp_17.keys)
    if key_resp_17.keys != None:  # we had a response
        thisExp.addData('key_resp_17.rt', key_resp_17.rt)
        thisExp.addData('key_resp_17.duration', key_resp_17.duration)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('selected_and_foils.csv'),
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "image_recall_task" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('image_recall_task.started', globalClock.getTime(format='float'))
        image_input.setImage(ImagePath)
        # create starting attributes for image_recall_response
        image_recall_response.keys = []
        image_recall_response.rt = []
        _image_recall_response_allKeys = []
        # Run 'Begin Routine' code from code_11
        start
        # keep track of which components have finished
        image_recall_taskComponents = [image_input, image_recall_response]
        for thisComponent in image_recall_taskComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "image_recall_task" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_input* updates
            
            # if image_input is starting this frame...
            if image_input.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_input.frameNStart = frameN  # exact frame index
                image_input.tStart = t  # local t and not account for scr refresh
                image_input.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_input, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_input.started')
                # update status
                image_input.status = STARTED
                image_input.setAutoDraw(True)
            
            # if image_input is active this frame...
            if image_input.status == STARTED:
                # update params
                pass
            
            # *image_recall_response* updates
            waitOnFlip = False
            
            # if image_recall_response is starting this frame...
            if image_recall_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_recall_response.frameNStart = frameN  # exact frame index
                image_recall_response.tStart = t  # local t and not account for scr refresh
                image_recall_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_recall_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_recall_response.started')
                # update status
                image_recall_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(image_recall_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(image_recall_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if image_recall_response.status == STARTED and not waitOnFlip:
                theseKeys = image_recall_response.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _image_recall_response_allKeys.extend(theseKeys)
                if len(_image_recall_response_allKeys):
                    image_recall_response.keys = _image_recall_response_allKeys[-1].name  # just the last key pressed
                    image_recall_response.rt = _image_recall_response_allKeys[-1].rt
                    image_recall_response.duration = _image_recall_response_allKeys[-1].duration
            # Run 'Each Frame' code from code_8
            //console.log(stimuli);
            
            for (var stimulus, _pj_c = 0, _pj_a = stimuli, _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
                 stimulus = _pj_a[_pj_c];
                 stimulus.draw(true);
                }
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in image_recall_taskComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "image_recall_task" ---
        for thisComponent in image_recall_taskComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('image_recall_task.stopped', globalClock.getTime(format='float'))
        # check responses
        if image_recall_response.keys in ['', [], None]:  # No response was made
            image_recall_response.keys = None
        trials_2.addData('image_recall_response.keys',image_recall_response.keys)
        if image_recall_response.keys != None:  # we had a response
            trials_2.addData('image_recall_response.rt', image_recall_response.rt)
            trials_2.addData('image_recall_response.duration', image_recall_response.duration)
        # the Routine "image_recall_task" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "small_break" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('small_break.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        small_breakComponents = [text_27]
        for thisComponent in small_breakComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "small_break" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_27* updates
            
            # if text_27 is starting this frame...
            if text_27.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_27.frameNStart = frameN  # exact frame index
                text_27.tStart = t  # local t and not account for scr refresh
                text_27.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_27, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_27.started')
                # update status
                text_27.status = STARTED
                text_27.setAutoDraw(True)
            
            # if text_27 is active this frame...
            if text_27.status == STARTED:
                # update params
                pass
            
            # if text_27 is stopping this frame...
            if text_27.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_27.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_27.tStop = t  # not accounting for scr refresh
                    text_27.tStopRefresh = tThisFlipGlobal  # on global time
                    text_27.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_27.stopped')
                    # update status
                    text_27.status = FINISHED
                    text_27.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in small_breakComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "small_break" ---
        for thisComponent in small_breakComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('small_break.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp_18
        key_resp_18.keys = []
        key_resp_18.rt = []
        _key_resp_18_allKeys = []
        # keep track of which components have finished
        break_2Components = [break_txt_final, key_resp_18]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 300.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_txt_final* updates
            
            # if break_txt_final is starting this frame...
            if break_txt_final.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_txt_final.frameNStart = frameN  # exact frame index
                break_txt_final.tStart = t  # local t and not account for scr refresh
                break_txt_final.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_txt_final, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_txt_final.started')
                # update status
                break_txt_final.status = STARTED
                break_txt_final.setAutoDraw(True)
            
            # if break_txt_final is active this frame...
            if break_txt_final.status == STARTED:
                # update params
                pass
            
            # if break_txt_final is stopping this frame...
            if break_txt_final.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_txt_final.tStartRefresh + 300-frameTolerance:
                    # keep track of stop time/frame for later
                    break_txt_final.tStop = t  # not accounting for scr refresh
                    break_txt_final.tStopRefresh = tThisFlipGlobal  # on global time
                    break_txt_final.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_txt_final.stopped')
                    # update status
                    break_txt_final.status = FINISHED
                    break_txt_final.setAutoDraw(False)
            
            # *key_resp_18* updates
            waitOnFlip = False
            
            # if key_resp_18 is starting this frame...
            if key_resp_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_18.frameNStart = frameN  # exact frame index
                key_resp_18.tStart = t  # local t and not account for scr refresh
                key_resp_18.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_18, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_18.started')
                # update status
                key_resp_18.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_18.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_18.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_18 is stopping this frame...
            if key_resp_18.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_18.tStartRefresh + 300-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_18.tStop = t  # not accounting for scr refresh
                    key_resp_18.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_18.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_18.stopped')
                    # update status
                    key_resp_18.status = FINISHED
                    key_resp_18.status = FINISHED
            if key_resp_18.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_18.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_18_allKeys.extend(theseKeys)
                if len(_key_resp_18_allKeys):
                    key_resp_18.keys = _key_resp_18_allKeys[-1].name  # just the last key pressed
                    key_resp_18.rt = _key_resp_18_allKeys[-1].rt
                    key_resp_18.duration = _key_resp_18_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_18.keys in ['', [], None]:  # No response was made
            key_resp_18.keys = None
        trials_2.addData('key_resp_18.keys',key_resp_18.keys)
        if key_resp_18.keys != None:  # we had a response
            trials_2.addData('key_resp_18.rt', key_resp_18.rt)
            trials_2.addData('key_resp_18.duration', key_resp_18.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-300.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_2'
    
    
    # --- Prepare to start Routine "end_exp" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_exp.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    end_expComponents = [thanks]
    for thisComponent in end_expComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_exp" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks* updates
        
        # if thanks is starting this frame...
        if thanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks.frameNStart = frameN  # exact frame index
            thanks.tStart = t  # local t and not account for scr refresh
            thanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanks.started')
            # update status
            thanks.status = STARTED
            thanks.setAutoDraw(True)
        
        # if thanks is active this frame...
        if thanks.status == STARTED:
            # update params
            pass
        
        # if thanks is stopping this frame...
        if thanks.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thanks.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                thanks.tStop = t  # not accounting for scr refresh
                thanks.tStopRefresh = tThisFlipGlobal  # on global time
                thanks.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thanks.stopped')
                # update status
                thanks.status = FINISHED
                thanks.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_expComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_exp" ---
    for thisComponent in end_expComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_exp.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
