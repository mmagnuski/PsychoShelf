# -*- coding: utf-8 -*-

from psychopy import visual, event, core, gui
import codecs, os
import numpy as np

# TODOs:
# [ ] - add orderPresented to dataframe
# [ ] - handle trial type (how?) 
# [ ] - choosing avatar       ??
# [ ] - choosing avatar name  ??

# note:
# getting box indices for image 2 on ImgList:
# idx = ImgList[1].inbox.idx
#
# or to avoid errors:
# bx = ImgList[1].inbox
# if bx:
#     idx = bx.idx


def rgb2psych(lst):
	# translates [255, 127, 0] rgb notation to
	# [1, 0, -1] psychopy style
	outlst = []

	for itm in lst:
		outlst.append((itm - 127) / 127.)
	return outlst

# consider adding ImList class...

class DragImList:
	img = []
	shelf = []

	def __init__(self, imlist, **kwargs):

		# set passed attribures
		for name, value in kwargs.items():
			setattr(self, name, value)	

		self.img = []
		for im in imlist:
			self.img.append(
				DragIm(
					win   = self.win,
					shelf = self.shelf,
					image = im
					))
	
	def contains(self, obj):

		# returns first image that contains
		for im in self.img:
			if im.contains(obj):
				return im
		return []
	
	def give_dragged(self):
		for im in self.img:
			if im.drag:
				return im
		return []

	def __getitem__(self, index):
		return self.img[index]

	def __iter__(self):
		return iter(self.img)


# class dragIm
class DragIm:

	win   = []
	image = []
	im    = []
	inbox = []
	shelf = []
	drag  = False
	mouse = []
	timer = []
	pickTime      = []
	pick2DropTime = []
	pickPos       = []
	dropPos       = []

	def __init__(self, **kwargs):

		# set passed attribures
		for name, value in kwargs.items():
			setattr(self, name, value)

		self.im = visual.ImageStim(self.win, image = self.image)
		
		# update box as to what image it holds:
		if self.inbox:
			self.inbox.hasim = self.im

	def place_in_box(self, box):
		
		# only if given box is empty?

		if self.inbox:
			self.inbox.hasim = []
		self.inbox = box
		box.hasim  = self.im

		# set position
		sz = self.im.size
		middlelim = np.mean(np.dstack((box.frontlim, box.backlim)), 
			axis = 2, dtype = int)
		self.im.pos = [np.mean(middlelim[:,0], dtype = int), 
					   box.frontlim[0,1] + sz[1]/2]
		# print 'New position:', self.im.pos
		# print 'placed in box', box.idx

	def click_drag(self, mouse, timer = []):

		if timer:
			self.pickTime = timer.getTime()
			timer.reset()
			self.timer = timer

		self.drag = True
		self.pickPos = mouse.getPos()
		self.mouse = mouse

		# release box:
		self.inbox.hasim = []

	def drop(self):

		# get time
		if self.timer: 
			self.pick2DropTime = self.timer.getTime()

		# get box and place in box:
		mousepos = self.mouse.getPos()
		bx = self.shelf.box_from_pos(mousepos)

		# check if box is not filled
		if bx and not bx.hasim:
			self.inbox = bx
		self.place_in_box(self.inbox)

		self.dropPos = mousepos
		self.drag = False
		self.mouse = []

	def contains(self, ob):
		return self.im.contains(ob)

	def draw(self):

		if self.drag and self.mouse:
			self.im.pos = self.mouse.getPos()

			# draw
			self.im.draw()


class Shelf:

	win = [] # how to get the main psychopy window?
	h = [20, 100]
	w = [40, 100]
	drawFromLeft = True
	drawOrder = [0, 1, 2, 3]
	left_corner = [100, 100]
	boxDims = [4, 4]
	boxes = []
	cover_matrix = []
	which_covered = []
	where_covers = 'back'

	# boxContains ? (should be in box class probably) - what image is there

	# whichBox -> transforms x, y pos to box reference or None if no such box

	def __init__(self, **kwargs):

		# set passed attribures
		for name, value in kwargs.items():
			setattr(self, name, value)

		self.update_covers()
		self.create_boxes()


	def update_covers(self):

		created_matr = False
		# check cover attributes
		if not self.cover_matrix:
			created_matr = True
			self.cover_matrix = np.zeros(self.boxDims, dtype = bool)

		if not self.which_covered and not created_matr:
			x, y = np.nonzero(self.cover_matrix)
			if x.shape == 0:
				self.which_covered = []
			else:
				self.which_covered = [[i,j] for i, j in zip(x, y)]
		elif self.which_covered:
			if isinstance(self.which_covered[0], list):
				ind = [[i[0] for i in self.which_covered],
						[i[1] for i in self.which_covered]]
			else:
				ind = self.which_covered

			self.cover_matrix[ind] = True


	def create_boxes(self):

		# create each row
		# boxes other than first in the row get [F,T,T,T]
		# boxes other than in the first row get [T,F,T,T]
		self.boxes = []
		for r in range(self.boxDims[0]):
			row = []
			for c in range(self.boxDims[1]):

				drw = [c == 0, r == 0, True, True]
				lc  = [self.left_corner[0] + c * self.w[1],
					   self.left_corner[1] + r * self.h[1]]
				cvr = self.cover_matrix[r,c]

				row.append(
					Box(
						h = self.h, 
						w = self.w,
						win = self.win, 
						left_corner = lc, 
						create_sides = drw,
						idx = [r, c],
						hascover = cvr,
						cover_side = self.where_covers,
						drawOrder = self.drawOrder
						)
					)

			self.boxes.append(row)


	def draw(self):
		if self.drawFromLeft:
			idx = range(0, self.boxDims[1])
		else:
			idx = range(self.boxDims[1] - 1, -1, -1)

		for row in self.boxes:
			for i in idx:
				row[i].draw()	


	def box_from_pos(self, pos):

		for row in self.boxes:
			for box in row:

				if box.frontlim[0,0] <= pos[0] and \
				   box.frontlim[1,0] >= pos[0] and \
				   box.frontlim[0,1] <= pos[1] and \
				   box.frontlim[1,1] >= pos[1]:
				   return box
		return []


	def clear(self):

		# clear images and covers from shelf
		for row in self.boxes:
			for bx in row:

				# clear images
				if bx.hasim:
					bx.hasim.inbox = []
					bx.hasim.shelf = []
					bx.hasim = []

				# clear covers
				if bx.hascover:
					bx.hascover = False
					bx.cover_shape = []

		self.cover_matrix = []
		self.which_covered = []


class Box:

	'''
	parameters:
	h - height, vector with two values, [height_change, full_height]
	w - width, vector with two values, [width_change, full_width]

	sides order:
	left, bottom, right, top
	'''

	win = [] # how to get the main psychopy window?
	hasim = []

	h = [20, 100]
	w = [40, 100]
	left_corner = [100, 100]
	
	vert  = []
	shape = []
	fillcolor = [[153, 102, 0], 
				 [204, 153, 0], 
				 [153, 102, 0], 
				 [204, 153, 0]]
	create_sides = [True, True, True, True]
	drawOrder = [0, 1, 2, 3]
	frontlim = [] # [xmin, xmax; ymin, ymax]
	backlim = [] 

	hascover = False
	cover_shape = []
	cover_side = 'back'
	cover_fillcolor = [102, 51, 0]
	idx = []

	def __init__(self, **kwargs):

		# set passed attribures
		for name, value in kwargs.items():
			setattr(self, name, value)

		# create limits
		self.backlim = \
			self.left_corner + \
			np.array([
				[0, 0], 
				[self.w[1], self.h[1] ] 
				])

		self.frontlim = self.backlim - [self.w[0], self.h[0]]

		# draw shapes
		self._chngmat()
		self._createshapes()

	def _chngmat(self):

		# create main change schemas
		self.chng = [0, 0, 0, 0]
		self.chng[0] = np.array([
			[0, 0],
			[0, self.h[1]],
			[-self.w[0], self.h[1] - self.h[0]],
			[-self.w[0], -self.h[0]]
			])
		
		self.chng[1] = np.array([
			[0, 0],
			[self.w[1], 0],
			[self.w[1] - self.w[0], -self.h[0]],
			[-self.w[0], -self.h[0]]
			])
		
		self.chng[2] = self.chng[0] + [self.w[1], 0]
		self.chng[3] = self.chng[1] + [0, self.h[1]]

	def _createshapes(self):
		
		# sds - which sides to draw
		sds, = np.nonzero(self.create_sides)
		self.vert = []
		for i, ch in enumerate(self.chng):
			if self.create_sides[i]:
				self.vert.append(self.left_corner + ch)

		self.shape = []
		for i, v in zip(sds, self.vert):
			self.shape.append(
				visual.ShapeStim(
					self.win, 
					lineWidth  = 1.5, 
					fillColor  = rgb2psych(self.fillcolor[i]), 
					lineColor  = [-1, -1, -1], 
					vertices   = v, 
					closeShape = True  ))

		# if we need to draw cover
		if self.hascover:
			lim = self.backlim if self.cover_side == 'back' else self.frontlim
			cov_vert = np.array([
				[lim[0][0], lim[0][1]],
				[lim[1][0], lim[0][1]],
				[lim[1][0], lim[1][1]],
				[lim[0][0], lim[1][1]]
				])

			self.cover_shape = visual.ShapeStim(
					self.win, 
					lineWidth  = 1.5, 
					fillColor  = rgb2psych(self.cover_fillcolor), 
					lineColor  = [-1, -1, -1], 
					vertices   = cov_vert, 
					closeShape = True  )


	def draw(self):

		# draw back if required
		if self.hascover and self.cover_side == 'back':
			self.cover_shape.draw()

		if (not self.hasim) or \
			(self.hascover and self.cover_side == 'front'):

			drw  = np.array(self.create_sides)
			order = np.argsort(np.array(self.drawOrder)[drw])

			for o in order:
				self.shape[o].draw()
		else:
			drw  = np.array(self.create_sides)
			order_orig = np.array(self.drawOrder)
			order = order_orig[drw]

			order_idx = np.argsort(order)
			drw = drw[order_orig]

			# first draw sides 0, 1
			s = 0
			for i in range(2):
				if drw[i]:
					self.shape[order_idx[s]].draw()
					s += 1

			# then draw image
			self.hasim.draw()

			# only then sides 2 and 3
			for i in range(2, 4):
				if drw[i]:
					self.shape[order_idx[s]].draw()
					s += 1

		# draw front if required
		if self.hascover and self.cover_side == 'front':
			self.cover_shape.draw()


# READ TRIALS defs:

# reading in trial specs:

# freader should test for existence
# return empty if trial not exist
def freader(fl, fun):
	with open(fl, 'r') as f:
		lst = []
		for l in f:
			fun(lst, l)
		return lst

def freader_encoding(fl, encd):
	f = codecs.open( fl, encoding = encd )
	return f.read()

def read_as_text(lst, line):
	if line[-1] == '\n':
		line = line[0:-1]
	lst.append(line)
	return lst

def read_as_indx(lst, line):
	if line[-1] == '\n':
		line = line[0:-1]

	vals = line.split(', ')
	vals = [int(i) for i in vals]
	lst.append(vals)
	return lst

def read_as_int(lst, line):
	if line[-1] == '\n':
		line = line[0:-1]

	val = int(line)
	lst.append(val)
	return lst

def read_trial(pth):
	trial = {}

	trial['n'] = int(pth[-2:])

	fl = os.path.join(pth, 'images.txt')
	trial['img_list'] = freader(fl, read_as_text)

	fl = os.path.join(pth, 'positions.txt')
	trial['pos_list'] = freader(fl, read_as_indx)

	fl = os.path.join(pth, 'covered.txt')
	trial['cov'] = freader(fl, read_as_indx)

	fl = os.path.join(pth, 'backimages.txt')
	trial['back_im'] = freader(fl, read_as_text)

	fl = os.path.join(pth, 'backpositions.txt')
	trial['back_im_pos'] = freader(fl, read_as_indx)

	fl = os.path.join(pth, 'correct_image.txt')
	trial['correctImage'] = freader(fl, read_as_text)[0]
	fl = os.path.join(pth, 'correct_position.txt')
	trial['correctPosition'] = freader(fl, read_as_indx)[0]

	fl = os.path.join(pth, 'instructions.txt')
	trial['instruct'] = freader_encoding(fl, 'utf8')

	fl = os.path.join(pth, 'instrtime.txt')
	trial['instructionTime'] = freader(fl, read_as_int)[0]

	return trial

def read_tutorial(pth):
	trial = {}

	trial['n'] = 0

	fl = os.path.join(pth, 'images.txt')
	trial['img_list'] = freader(fl, read_as_text)

	fl = os.path.join(pth, 'positions.txt')
	trial['pos_list'] = freader(fl, read_as_indx)

	fl = os.path.join(pth, 'covered.txt')
	trial['cov'] = freader(fl, read_as_indx)

	fl = os.path.join(pth, 'backimages.txt')
	trial['back_im'] = freader(fl, read_as_text)

	fl = os.path.join(pth, 'backpositions.txt')
	trial['back_im_pos'] = freader(fl, read_as_indx)

	fl = os.path.join(pth, 'correct_image.txt')
	trial['correctImage'] = freader(fl, read_as_text)[0]
	fl = os.path.join(pth, 'correct_position.txt')
	trial['correctPosition'] = freader(fl, read_as_indx)[0]

	for i in range(1, 8):
		fl = os.path.join(pth, 'instructions' + \
			fillz(i, 2) + '.txt')
		trial['instruct' + str(i)] = freader_encoding(fl, 'utf8')

	#fl = os.path.join(pth, 'instrtime.txt')
	#trial['instructionTime'] = freader(fl, read_as_int)[0]

	return trial


def read_all_trials(exp):

	# check how many trials:
	trialpath = os.path.join(exp['pth'], 'trials')
	lst = os.listdir(trialpath)

	trials = []
	for t in lst:
		if t[0:5] == 'trial':
			pth = os.path.join(trialpath, t)
			trials.append(read_trial(pth))

	exp['numTrials'] = len(trials)
	return trials

# TRIAL helpers:
# --------------

def fillTrial(trial, im, db):
	tr = trial['n']
	db.loc[tr, 'orderPresented'] = trial['order']
	db.loc[tr, 'objMoved']  = im.image
	db.loc[tr, 'movedTo']   = str(im.inbox.idx)
	db.loc[tr, 'isCorrect'] = im.image == trial['correctImage'] and \
							  im.inbox.idx == trial['correctPosition']
	db.loc[tr, 'pickRT']        = im.pickTime
	db.loc[tr, 'pick2dropTime'] = im.pick2DropTime
	db.loc[tr, 'startMousePos'] = str(trial['startPos'])
	db.loc[tr, 'pickMousePos']  = str(im.pickPos)
	db.loc[tr, 'dropMousePos']  = str(im.dropPos)

def checkTrial(trial, im):
	correct = \
		im.image == trial['correctImage'] and \
		im.inbox.idx == trial['correctPosition']
	return correct


def add_fname(trial):
	# add images folder name:
	ims = ['img_list', 'back_im']
	for k in ims:
		for i in range(len(trial[k])):
			trial[k][i] = os.path.join('images', 
				trial[k][i])
	trial['correctImage'] = os.path.join('images', 
		trial['correctImage'])

def run_trial(trial, exp, win, mouse, db):

	trial['startPos'] = mouse.getPos()

	# correct folder names:
	add_fname(trial)

	# create stimuli
	shelf, imgs, bckg, instr = createstims(win, trial)
	# we want to add background to instructions
	instrback = addback(win, instr)
	bckg = addgeomback(win, bckg)
	
	# get middle shelf pos:
	midpos = shelf.boxes[2][2].frontlim
	midpos = [midpos[0,0], midpos[0,1]]

	# create dot
	dot = visual.Circle(win, 
		radius = 20,
		edges = 32,
		pos = midpos,
		fillColor = [-0.7, -0.7, -0.7])

	# countdown obj:
	countdown = visual.TextStim(
		win, 
		text = '3', 
		pos = midpos,
		height = 20,
		color = [0.5, 0.5, 0.5])

	# do not hide mouse
	win.setMouseVisible(True)

	# 1. draw backim and instructions
	timer = core.CountdownTimer(trial['instructionTime'])
		
	while timer.getTime() > 0:
		for b in bckg:
			b.draw()

		# instructions remain on screen
		instrback.draw()
		instr.draw()

		# flip the window hooray
		win.flip()


	# show mouse 
	win.setMouseVisible(True)
	# mouse.setPos(newPos = [0, 0])

	instr.setText(u'Umieść myszkę na kropce')
	notOnDot = True
	while notOnDot:

		notOnDot = not testMousePos(mouse, midpos, 15)

		if notOnDot:
			for b in bckg:
				b.draw()
			instr.draw()
			dot.draw()
			win.flip()

	keepOnDot = False
	while not keepOnDot:
		for i in range(3, 0, -1):
			# set timer
			timer = core.CountdownTimer(0.75)
			countdown.setText(str(i))

			while timer.getTime() > 0:
				
				# draw
				for b in bckg:
					b.draw()
				instr.draw()
				dot.draw()
				countdown.draw()
				win.flip()

				# test if on dot
				keepOnDot = testMousePos(mouse, midpos, 15)
				if not keepOnDot:
					break
		
			if not keepOnDot:
				break

	# create timer
	tmr = core.Clock()

	# 2. trial drag-drop loop
	# -----------------------
	runTrial = True
	while runTrial:

		# handle quit keypresses
		for key in event.getKeys():
			if key in ['escape','q']:
				core.quit()

		m1, m2, m3 = mouse.getPressed()

		# test for drag-and-drop
		dragged = imgs.give_dragged()
		if m1 and not dragged:
			im = imgs.contains(mouse)

			# diagnostics0(imgs, im, shelf)

			if im:
				im.click_drag(mouse, timer = tmr)

		elif not m1 and dragged:
			# drop object
			dragged.drop()

			# fill database
			fillTrial(trial, dragged, db)

			# finish trials
			runTrial = False;
			
			# wasdragged = dragged
			# dragged = []
			# diagnostics(dragged, shelf)

		# drawing
		drawstims(bckg, shelf, dragged, win)

	# database is written to disc after every trial
	outfl = os.path.join(exp['pth'], 'behdata', exp['participant'] + '.xls')
	db.to_excel( outfl )


def run_tutorial(trial, exp, win, mouse):

	trial['type'] = 'tutorial'
	trial['instruct'] = trial['instruct1']

	# correct folder names:
	add_fname(trial)

	# create stimuli
	shelf, imgs, bckg, instr = createstims(win, trial)
	# we want to add background to instructions
	bckg = addgeomback(win, bckg)
	
	# get middle shelf pos:
	midpos = shelf.boxes[2][2].frontlim
	midpos = [midpos[0,0], midpos[0,1]]

	# create dot
	dot = visual.Circle(win, 
		radius = 20,
		edges = 32,
		pos = midpos,
		fillColor = [-0.7, -0.7, -0.7])

	# countdown obj:
	countdown = visual.TextStim(
		win, 
		text = '3', 
		pos = midpos,
		height = 20,
		color = [0.5, 0.5, 0.5])

	# do not hide mouse
	win.setMouseVisible(True)

	# display instructions 01 - 02

	dragged = []
	for i in range(1, 3):
		# clear buffer
		m1, m2, m3 = mouse.getPressed()
		# setup
		timer = core.CountdownTimer(1.5)
		instr.setText(trial['instruct' + str(i)])
		instrback = addback(win, instr, op = 0.7)
		
		while True:
			drawstims(bckg, shelf, dragged, 
				win, noflip = True)
			instrback.draw()
			instr.draw()
			win.flip()
			if timer.getTime() < 0:
				m1, m2, m3 = mouse.getPressed()
				if m1:
					break

	# reversed shelf, special case:
	# -----------------------------
	trial['instruct'] = trial['instruct3']

	# get other view
	trial2 = flip_shelf(trial)
	shelf2, imgs2, b, instr = createstims(
		win, trial2, rev_shelf = True,
		lc = [-250, -250])

	# put images to boxes
	for i, im in zip(trial2['pos_list'], imgs2):
		bx = shelf2.boxes[i[0]][i[1]]
		im.place_in_box(bx)

	# get back for instructions
	instrback = addback(win, instr, op = 0.7)
	
	# new geom back
	bckg2 = []
	bckg2 = addgeomback(
		win, bckg2, 
		prop = 1,
		start = [410, -250], 
		xchng = -200,
		inv = True)
		
	while True:
		for b in bckg2:
			b.draw()
		shelf2.draw()
		instrback.draw()
		instr.draw()
		win.flip()

		m1, m2, m3 = mouse.getPressed()
		if m1:
			break

	# clean up
	del bckg2
	del shelf2
	del trial2


	# instr 4 
	# -------

	# clear buffer
	m1, m2, m3 = mouse.getPressed()
	# setup
	instr.setText(trial['instruct' + str(4)])
	instrback = addback(win, instr, op = 0.7)
	timer = core.CountdownTimer(1.5)
	
	while True:
		drawstims(bckg, shelf, dragged, 
			win, noflip = True)
		instrback.draw()
		instr.draw()
		win.flip()

		if timer.getTime() < 0:
			m1, m2, m3 = mouse.getPressed()
			if m1:
				break


	# instr 5 - dragging
	# ------------------

	correct = False
	while not correct:

		# clear buffer
		m1, m2, m3 = mouse.getPressed()

		# set up instructions
		instr.setText(trial['instruct' + str(5)])
		instr.pos = [0, 260]
		instrback = addback(win, instr, op = 0.7)

		runTrial = True
		tmr = core.Clock()
		while runTrial:

			m1, m2, m3 = mouse.getPressed()

			# test for drag-and-drop
			dragged = imgs.give_dragged()
			if m1 and not dragged:
				im = imgs.contains(mouse)

				# diagnostics0(imgs, im, shelf)

				if im:
					im.click_drag(mouse, timer = tmr)

			elif not m1 and dragged:
				# drop object
				dragged.drop()

				# check correctness
				correct = checkTrial(trial, dragged)

				# finish trials
				runTrial = False;
				
				# wasdragged = dragged
				# dragged = []
				# diagnostics(dragged, shelf)

			# drawing
			drawstims(bckg, shelf, dragged, win, noflip = True)
			instrback.draw()
			instr.draw()
			win.flip()

		if not correct:
			instr.setText(u'Źle, spróbuj jeszcze raz.')
			drawstims(bckg, shelf, dragged, win, noflip = True)
			instrback.draw()
			instr.draw()
			win.flip()
			core.wait(2)

			# return items to original shelves:
			for i, im in zip(trial['pos_list'], imgs):
				bx = shelf.boxes[i[0]][i[1]]
				im.place_in_box(bx)

	# instructions 06 - 07
	# --------------------

	for i in range(6, 8):

		# clear buffer
		m1, m2, m3 = mouse.getPressed()
		# setup
		instr.setText(trial['instruct' + str(i)])
		instr.pos = [0, 245]
		instrback = addback(win, instr, op = 0.85)
		timer = core.CountdownTimer(1.5)

		while True:
			drawstims(bckg, shelf, dragged, 
				win, noflip = True)
			instrback.draw()
			instr.draw()
			win.flip()

			if timer.getTime() < 0:
				m1, m2, m3 = mouse.getPressed()
				if m1:
					break


def createstims(win, trial, lc = [-300, -250], \
	rev_shelf = False):
	
	# create shelf
	if not rev_shelf:
		shelf = Shelf(
			win = win, 
			left_corner = lc,
			which_covered = trial['cov'])
	else:
		shelf = Shelf(
			win = win, 
			h = [20, 100],
			w = [-20, 100],
			left_corner = lc,
			drawFromLeft = False,
			drawOrder = [2, 0, 1, 3],
			which_covered = trial['cov'],
			where_covers = 'front')

	# create images:
	imgs = DragImList(
		trial['img_list'], 
		win = win, 
		shelf = shelf)

	# create text object:
	instr = visual.TextStim(
		win, 
		text = trial['instruct'], 
		pos = [0, 245],
		height = 25)

	# create backgroud image:
	bckg  = []
	for im, pos in zip(trial['back_im'], trial['back_im_pos']):
		bckg.append(
			visual.ImageStim(
				win, 
				image = im,
				pos = pos))

	# put images in respective boxes:
	for i, im in zip(trial['pos_list'], imgs):
		bx = shelf.boxes[i[0]][i[1]]
		im.place_in_box(bx)

	return shelf, imgs, bckg, instr


def drawstims(bckg, shelf, dragged, win, noflip = False):
	# first - draw background
	for b in bckg:
		b.draw()

	# draw shelf
	shelf.draw()

	# draw dragged img if should be dragging
	if dragged:
		dragged.draw()

	# flip the window hooray
	if not noflip:
		win.flip()

def testMousePos(mouse, pos, tol):
	# get mouse position
	mpos = mouse.getPos()
	# test with respect to goal position
	if np.abs(mpos[0] - pos[0]) <= tol and \
		np.abs(mpos[1] - pos[1]) <= tol:
		return True
	else:
		return False

def flip_shelf(trial):
	# flip image positions (assume 4 by 4 nDims):
	t = trial.copy()

	# flip item and cover positions
	flds = ['pos_list', 'cov']
	for f in flds:
		t[f] = [ [i, 3 - j] for i, j in t[f]]
	return t

def addgeomback(win, bckg, prop = 0.5,
	start = [-410, -250], xchng = 245,
	inv = False):
	# prop is just the slope of the line
	# (y / x)
	newbckg = []

	# check for inversion
	if not inv:
		sgn = -1
	else:
		sgn = 1

	# floor
	screenlim = [410, 310]
	ychng = int(xchng * prop * (sgn * -1))
	topnt = [start[0] + xchng, start[1] + ychng]

	vert = [start, 
			topnt,
			[start[0]*-1, topnt[1]], 
			[start[0]*-1, start[1]],
			[start[0]*-1, screenlim[1]*-1], 
			[start[0], screenlim[1]*-1]]
	floor_col = rgb2psych([158, 191, 113])
	# create shape
	newbckg.append(
		shape(win, vert, floor_col)
		)

	# left wall
	vert = [start, 
			topnt,
			[topnt[0], screenlim[1]], 
			[screenlim[0]*sgn, screenlim[1]]]
	wall_col = rgb2psych([178, 178, 178])
	# create shape
	newbckg.append(
		shape(win, vert, wall_col)
		)

	return newbckg + bckg

def shape(win, vert, col, lw = 1.5, op = 1):
	return visual.ShapeStim(
		win,
		lineWidth  = lw, 
		vertices   = vert, 
		fillColor  = col, 
		lineColor  = [-1, -1, -1], 
		closeShape = True,
		opacity    = op  
		)

def addback(win, instr, lim = 15, op = 0.5):
	# get instr position
	pos = instr.pos

	# if goes above screen - correct:
	h = int(instr.height / 2) + lim
	w = int(instr.width / 2) + lim
	dst = 300 - (pos[1] + h)
	if dst < 0:
		pos[1] = pos[1] + dst
		instr.setPos(pos)

	lims = pos + np.array([[-w,-h], [-w,h], [w,h], [w,-h]]) 
	return shape(win, lims, [-0.5, -0.5, -0.5], op = op)


def fillz(val, num):
    '''fillz(val, num)
    adds zero to the beginning of val so that length of
    val is equal to num. val can be string, int or float'''
    
    # if not string - turn to a string
    if not isinstance(val, basestring):
        val = str(val)
     
    # add zeros
    ln = len(val)
    if ln < num:
        return '0' * (num - ln) + val
    else:
        return val

def diagnostics(im, shelf):

	print 'Dropped image is in box', im.inbox.idx
	print '(should be in [1, 1])'
	print 'This box is:', im.inbox
	print 'The property hasim of the box is:'
	if im.inbox.hasim:
		print 'filled'
	else:
		print 'empty'

	print 'shelf.boxes[1][1]:', shelf.boxes[1][1]
	print 'shelf.boxes[1][1].hasim:'
	if shelf.boxes[1][1].hasim:
		print 'filled'
	else:
		print 'empty'
	print 'shelf.boxes[1][1].idx:', shelf.boxes[1][1].idx

	print 'if shelf and image.shelf are the same'
	print shelf == im.shelf

def diagnostics0(imgs, im, shelf):
	print '\n'
	print 'Just grabbed some image!'
	print 'Its shelf is identical to main shelf:'
	print im.shelf == shelf

	print 'imgs.shelf == shelf ?'
	print imgs.shelf == shelf

	print 'Which im from imgs has shelf?'
	testshe = []
	for i in imgs:
		testshe.append(i.shelf == shelf)

	print testshe

	print 'what images are in imgs:'
	for i in imgs:
		print i.image

# get user name:
def GetUserName():
    '''
    PsychoPy's simple GUI for entering 'stuff'
    Does not look nice or is not particularily user-friendly
    but is simple to call.
    Here it just asks for subject's name/code
    and returns it as a string
    '''
    myDlg = gui.Dlg(title="Osoba badana", size = (800,600))
    myDlg.addText('Podaj kod osoby badanej')
    myDlg.addField('kod: ')
    myDlg.show()  # show dialog and wait for OK or Cancel

    if myDlg.OK:  # the user pressed OK
        dialogInfo = myDlg.data
        user = dialogInfo[0]
    else:
        user = 'Anonymous' + str(randint(0, 1000))
   
    return user