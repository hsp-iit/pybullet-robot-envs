#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

log_dir = '../pybullet_logs/icubreach_deepq'

def main():

    results = pu.load_results(log_dir)

    if not results:
        return

    r = results[0]
    plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
    plt.show()

if __name__ == '__main__':
  main()
