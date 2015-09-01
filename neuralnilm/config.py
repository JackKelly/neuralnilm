from __future__ import print_function, division
import os
import ConfigParser

config = ConfigParser.RawConfigParser()
filename = os.path.expanduser('~/.neuralnilm')
config.read(filename)
