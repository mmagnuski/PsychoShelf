# -*- coding: utf-8 -*-
from   psychopy   import visual, event, core
import os
import numpy        as   np
import pandas       as   pd
import shelfutils   as   sh
print 'imports finished'

# settings
doTutorial = True
doFirstFour = True

# create window, get mouse
win   = visual.Window([800, 600], units = 'pix')
# , winType = 'pygame'  - pygame not installed or not fully working
mouse = event.Mouse()

# exp data and dataframe
exp = {}
# check script path
exp['pth'] = os.path.dirname(os.path.abspath(__file__))
# CHANGE:
exp['participant'] = sh.GetUserName()

# read in all trials:
trials = sh.read_all_trials(exp)

# create dataframe to hold results
colNames = ['orderPresented', 'objMoved', 'movedTo', 
			'isCorrect', 'pickRT', 'pick2dropTime',
			'startMousePos', 'pickMousePos', 'dropMousePos']
db = pd.DataFrame(index = np.arange(1, exp['numTrials'] + 1), \
				  columns = colNames )

# TUTORIAL
if doTutorial:
	tut = sh.read_tutorial(os.path.join(exp['pth'], 'trials', 'tutorial'))
	sh.run_tutorial(tut, exp, win, mouse)


# TRIAL PART
# ----------
o = 1
if doFirstFour:
	for t in range(1, 5):

		# run single trial:
		trials[t-1]['order'] = o
		o += 1
		sh.run_trial(trials[t-1], exp, win, mouse, db)
		core.wait(0.5)

# randomize all other trials:
numt = len(trials)
tri  = np.arange(5, numt + 1)
np.random.shuffle(tri)

for t in tri:

	# run single trial:
	trials[t-1]['order'] = o
	o += 1
	sh.run_trial(trials[t-1], exp, win, mouse, db)
	core.wait(0.5)

# at the end - quit
core.quit()