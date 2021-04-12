import os
import gym
import gym_pygame.envs.flappybird

current_path = os.path.dirname(__file__) # Where your .py file is located
resource_path = os.path.join(current_path, 'resources') # The resource folder path
image_path = os.path.join(resource_path, 'images') #

env = gym.make("FlappyBird-PLE-v0")
done = False

while not done:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()


exit()

from dataset.gym_client_creator import GymClientCreator

e = GymClientCreator().create_flappy_bird()
e = e.get_environment()[0]


while True:
    e.test_env()

exit()

env = gym.make('CarRacing-v0')
env.reset()

print(env.action_space)
print(env.observation_space)

for _ in range(1000):
    # env.render()
    env.step(env.action_space.sample()) # take a random action

env.close()
