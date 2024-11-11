#!/usr/bin/env python3
#encoding: utf-8

import matplotlib.pyplot as plt
from collections import deque

class dispTrace:
   def __init__(self, tr=True, trLn=10):
      self.fig = plt.figure()
      self.ax = self.fig.add_subplot(autoscale_on=False)
      self.nTraces = 0
      self.traceLen = trLn
      self.points = []
      self.traces = []
      self.queues = []
   def setPoints(self, frmt):
      self.points = []
      for f in frmt:
         self.points.append(self.ax.plot([], [], f)[0])
   def setTraces(self, frmt, trLen = None):
      self.traces = []
      self.queues = []
      idx = 0
      for f in frmt:
         self.traces.append(self.ax.plot([], [], f, lw = 1, ms = 2)[0])
         if trLen != None:
            self.queues += [deque(maxlen = trLen[idx]), deque(maxlen = trLen[idx])]
            idx += 1
         else:
            self.queues += [deque(maxlen = self.traceLen), deque(maxlen = self.traceLen)]
      self.nTraces = len(self.traces)
   def setLimits(self, xMin, xMax, yMin, yMax, fact=1.2):
      xmd  = (xMin + xMax) / 2
      ymd  = (yMin + yMax) / 2
      dx2 = (xMax - xMin) / 2
      dy2 = (yMax - yMin) / 2
      xm, xM = xmd - fact * dx2, xmd + fact * dx2
      ym, yM = ymd - fact * dy2, ymd + fact * dy2
      self.ax.set_xlim(xm, xM)
      self.ax.set_ylim(ym, yM)
