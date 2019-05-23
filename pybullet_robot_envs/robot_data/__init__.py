import os


def getDataPath():
  resdir = os.path.join(os.path.dirname(__file__))
  print("resdir ",resdir)
  return resdir
