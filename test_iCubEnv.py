
from envs.iCubGymEnv import iCubGymEnv
import time

def main():

    env = iCubGymEnv(renders=True)

    env.reset()

    for t in range(100000):
        ##  To obtain images and depth
        ##  env.render()
        env.step()

if __name__ == '__main__':
    main()
