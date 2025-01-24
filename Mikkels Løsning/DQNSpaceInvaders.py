# %% Load libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import mdock
import SpaceInvaders
import pygame

# %% Parameters
n_games = 10000
epsilon = 1
epsilon_min = 0.01
epsilon_reduction_factor = 0.01**(1/10000)
gamma = 0.99
batch_size = 512
buffer_size = 100_000
learning_rate = 0.0001
steps_per_gradient_update = 10
max_episode_step = 100
input_dimension = 51
hidden_dimension = 256
output_dimension = 4

# %% Neural network, optimizer and loss
q_net = torch.nn.Sequential(
    torch.nn.Linear(input_dimension, hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dimension, hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dimension, output_dimension)
)
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()

# %% State to input transformation
# Convert environment state to neural network input by one-hot encoding the state
def state_to_input(state):
    player_x, enemy_x, enemy_y, enemy_direction, bullet_x, bullet_y, bullet_on = state
    player_x_onehot = torch.nn.functional.one_hot(torch.tensor(player_x), num_classes=10)
    enemy_x_onehot = torch.nn.functional.one_hot(torch.tensor(enemy_x), num_classes=10)
    enemy_y_onehot = torch.nn.functional.one_hot(torch.tensor(enemy_y), num_classes=10)
    enemy_direction_onehot = torch.tensor([0.]) if enemy_direction==-1 else torch.tensor([1.])
    if bullet_on:
        bullet_x_onehot = torch.nn.functional.one_hot(torch.tensor(bullet_x), num_classes=10)
        bullet_y_onehot = torch.nn.functional.one_hot(torch.tensor(bullet_y), num_classes=10)
    else:
        bullet_x_onehot = bullet_y_onehot = torch.zeros(10)
    return torch.hstack((player_x_onehot, enemy_x_onehot, enemy_y_onehot, enemy_direction_onehot, bullet_x_onehot, bullet_y_onehot))

# %% Environment
env = SpaceInvaders.SpaceInvaders()
action_names = env.actions        # Actions the environment expects
actions = np.arange(4)            # Action numbers

# %% Buffers
obs_buffer = torch.zeros((buffer_size, input_dimension))
obs_next_buffer = torch.zeros((buffer_size, input_dimension))
action_buffer = torch.zeros(buffer_size).long()
reward_buffer = torch.zeros(buffer_size)
done_buffer = torch.zeros(buffer_size)

# %% Training loop

# Logging
scores = []
losses = []
episode_steps = []
step_count = 0
print_interval = 100

# Training loop
for i in range(n_games):
    # Reset game
    score = 0
    episode_step = 0
    episode_loss = 0
    episode_gradient_step = 0
    done = False
    env_observation = env.reset()
    observation = state_to_input(env_observation)

    # Reduce exploration rate
    epsilon = (epsilon-epsilon_min)*epsilon_reduction_factor + epsilon_min
    
    # Episode loop
    while (not done) and (episode_step < max_episode_step):       
        # Choose action and step environment
        if np.random.rand() < epsilon:
            # Random action
            action = np.random.choice(actions)
        else:
            # Action according to policy
            action = np.argmax(q_net(observation).detach().numpy())
        env_observation_next, reward, done = env.step(action_names[action])
        observation_next = state_to_input(env_observation_next)
        score += reward

        # Store to buffers
        buffer_index = step_count % buffer_size
        obs_buffer[buffer_index] = observation
        obs_next_buffer[buffer_index] = observation_next
        action_buffer[buffer_index] = action
        reward_buffer[buffer_index] = reward
        done_buffer[buffer_index] = done

        # Update to next observation
        observation = observation_next

        # Learn using minibatch from buffer (every steps_per_gradient_update)
        if step_count > batch_size and step_count%steps_per_gradient_update==0:
            # Choose a minibatch            
            batch_idx = np.random.choice(np.minimum(
                buffer_size, step_count), size=batch_size, replace=False)

            # Compute loss function
            out = q_net(obs_buffer[batch_idx])
            val = out[np.arange(batch_size), action_buffer[batch_idx]]   # Explain this indexing
            with torch.no_grad():
                out_next = q_net(obs_next_buffer[batch_idx])
                target = reward_buffer[batch_idx] + \
                    gamma*torch.max(out_next, dim=1).values * \
                    (1-done_buffer[batch_idx])
            loss = loss_function(val, target)

            # Step the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_gradient_step += 1
            episode_loss += loss.item()

        # Update step counteres
        episode_step += 1
        step_count += 1

    scores.append(score)
    losses.append(episode_loss / (episode_gradient_step+1))
    episode_steps.append(episode_step)
    if (i+1) % print_interval == 0:
        # Print average score and number of steps in episode
        average_score = np.mean(scores[-print_interval:-1])
        average_episode_steps = np.mean(episode_steps[-print_interval:-1])
        print(f'Episode={i+1}, Score={average_score:.1f}, Steps={average_episode_steps:.0f}')

        # Plot scores        
        plt.figure('Score')
        plt.clf()
        plt.plot(scores, '.')
        plt.title(f'Step {step_count}: eps={epsilon:.3}')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        
        # Plot number of steps
        plt.figure('Steps per episode')
        plt.clf()
        plt.plot(episode_steps, '.')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)

        # Plot last batch loss
        plt.figure('Loss')
        plt.clf()
        plt.plot(losses, '.')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)

        # Estimate validation error

        mdock.drawnow()

        # Save model
        # torch.save(q_net.state_dict(), 'q_net.pt')
env.close()

# %% Play loop
# Load model
q_net.load_state_dict(torch.load('q_net.pt'))

# Create envionment
env = SpaceInvaders.SpaceInvaders()

# Reset game
score = 0
done = False
observation = env.reset()
episode_step = 0

# Play episode        
with torch.no_grad():    
    while (not done) and (episode_step < max_episode_step):
        pygame.event.get()
        # Choose action and step environment
        action = np.argmax(q_net(state_to_input(observation)).detach().numpy())
        observation, reward, done = env.step(action_names[action])
        score += reward
        env.render()
        episode_step += 1

# Print score
print(f'Score={score:.0f}')

# Close and clean up
env.close()
