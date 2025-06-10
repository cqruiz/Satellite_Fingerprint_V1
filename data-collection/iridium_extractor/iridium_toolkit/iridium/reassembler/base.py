#!/usr/bin/env python3
# vim: set ts=4 sw=4 tw=0 et pm=:

import sys
import datetime
import math

from ..config import config

if sys.version_info[0]==3 and sys.version_info[1]<8:
    print("Old python detected, using replacement bytes class...", file=sys.stderr)
    from util import mybytes
    globals()['bytes']=mybytes

base_freq=1616*10**6
channel_width=41667

class Zulu(datetime.tzinfo):
    def utcoffset(self, dt):
        return datetime.timedelta(0)
    def dst(self, dt):
        return datetime.timedelta(0)
    def tzname(self,dt):
         return "Z"

Z=Zulu()

pwarn=False

class MyObject(object):
    def enrich(self, channelize=False):
        if "|" in self.frequency:
            chan, off=self.frequency.split('|')
            self.frequency=base_freq+channel_width*int(chan)+int(off)
        else:
            self.frequency=int(self.frequency)

        if channelize:
            fbase=self.frequency-base_freq
            self.freq_chan=int(fbase/channel_width)
            foff =fbase%channel_width
            self.freq_off=foff-(channel_width/2)
            self.freq_print="%3d|%+06d"%(self.freq_chan,self.freq_off)

        if len(self.name) > 3 and self.name[1]=='-':
            self.ftype=self.name[0]
            self.starttime, _, self.attr = self.name[2:].partition('-')
        else:
            self.ftype = self.starttime = self.attr = ''

        self.confidence=int(self.confidence.strip("%"))
        self.mstime=float(self.mstime)

        if '|' in self.level:
            self.level, self.noise, self.snr = self.level.split('|')
            self.snr = float(self.snr)
            self.noise = float(self.noise)
            self.level=float(self.level)
        else:
            self.snr=None
            self.noise=None
            if float(self.level)==0:
                self.level+="1"
            try:
                self.level=math.log(float(self.level),10)*20
            except ValueError:
                print("Invalid signal level:",self.level, file=sys.stderr)
                self.level=0

        if self.ftype=='p':
            self.time=float(self.starttime)+self.mstime/1000
        elif self.ftype=='j': # deperec
            self.time=self.mstime
            self.timens=int(self.mstime*(10**9))
        else:
            try:
                # XXX: Does not handle really old time format.
                self.time=float(self.starttime)+self.mstime/1000
            except ValueError:
                self.time=self.mstime/1000

        if self.attr.startswith("e"):
            if self.attr != 'e000':
                self.perfect=False
            else:
                self.perfect=True
        else:
            if self.attr == 'UW:0-LCW:0-FIX:00':
                self.perfect=True
            else:
                self.perfect=False
            if 'perfect' in config.args:
                global pwarn
                if pwarn is False:
                    pwarn = True
                    print("'perfect' requested, but no EC info found", file=sys.stderr)

class Reassemble(object):
    def __init__(self):
        raise Exception("undef")
    stat_line=0
    stat_filter=0
    def run(self,producer):
        for line in producer:
            res=self.filter(line)
            if res != None:
                self.stat_filter+=1
                zz=self.process(res)
                if zz != None:
                    for mo in zz:
                        self.consume(mo)
        self.end()
    def filter(self,line):
        self.stat_line+=1
        try:
            q=MyObject()
            q.typ,q.name,q.mstime,q.frequency,q.confidence,q.level,q.symbols,q.uldl,q.data=line.split(None,8)
            return q
        except ValueError:
            print("Couldn't parse input line: ",line, end=' ', file=sys.stderr)
            return None

    def end(self):
        if self.stat_line>0:
            print("Kept %d/%d (%3.1f%%) lines"%(self.stat_filter,self.stat_line,100.0*self.stat_filter/self.stat_line))
        else:
            print("No lines?")

modes=[]
