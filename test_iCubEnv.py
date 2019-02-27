
from envs.iCubGymEnv import iCubGymEnv

def main():

    env = iCubGymEnv(renders=True)

    env.reset()

    for t in range(1000):
        env.render()
        env.step()

if __name__ == '__main__':
    main()
