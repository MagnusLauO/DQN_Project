import torch
import numpy as np
import matplotlib.pyplot as plt
from BallsFloat import BallsGame
import time
import os
import csv

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Parameters (kept as close as possible to original)
Episodes = 1000
Epsilon_max = 1
Epsilon_min = 0.01
Epsilon_DecayRate = 0.993109 # 
Discount_rate = 0.99
Batch_size = 512
Buffer_size = 100000
LR = 0.0001 # learning rate
Gradient_update = 10
State_dim = 4  # Modified for new state space
Hidden_dim = 128
Action_dim = 3  # Modified for new action space
Number = 0
max_steps = 5000
###################
# Neural network #
###################

NN = torch.nn.Sequential( 
    torch.nn.Linear(State_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Action_dim)
)
optimizer = torch.optim.Adam(NN.parameters(), lr=LR)
loss_function = torch.nn.MSELoss()

NN = NN.to(device)

Current_Network = 'NN2'

#####################
# States til onehot #
#####################


# Kunne laves om så ball_x er hele vejen når ball_exists er 0.
def state_to_input1(state):
    player_x, ball_x, ball_exists = state
    player_x_onehot = torch.nn.functional.one_hot(torch.tensor(player_x), num_classes=5)
    ball_x_onehot = torch.nn.functional.one_hot(torch.tensor(ball_x), num_classes=5)
    ball_exists_onehot = torch.tensor([float(ball_exists)])
    return torch.hstack((player_x_onehot, ball_x_onehot, ball_exists_onehot))


def state_to_input2(state):
    state = torch.tensor(state.flatten(), dtype=torch.float32)
    return state

def state_to_input3(state):
    state = torch.tensor(state, dtype=torch.float32)
    return state



##################
# Train function #
##################

def train():
    start_time = time.time()
    #########################
    # Environment + actions #
    #########################

    env = BallsGame()
    action_names = env.actions # left right none
    actions = np.arange(3)  # (0, 1, 2)


    ###########
    # Buffere #
    ###########

    # Kig mere på disse
    obs_buffer = torch.zeros((Buffer_size, State_dim),device=device)
    obs_next_buffer = torch.zeros((Buffer_size, State_dim),device=device)
    action_buffer = torch.zeros(Buffer_size,device=device).long()
    reward_buffer = torch.zeros(Buffer_size,device=device)
    done_buffer = torch.zeros(Buffer_size,device=device)



    # Data storage #
    GameRewards = []
    losses = []
    episode_steps = []
    step_count = 0
    print_interval = 100
    Epsilon = Epsilon_max

    ####################
    # Create directory #
    ####################
    # Help from ChatGPT

    agent_dir = f"Gus1_Agents/agent_Float_{str(Current_Network)}_{Hidden_dim}_{Number}_{Episodes}_{time.strftime('%Y-%m-%d_%H-%M-%S')}" # Directory with date and time in folder called Agents
    os.makedirs(agent_dir, exist_ok=True)

    # Save parameters in a text file
    params = {
        "Episodes": Episodes,
        "Learning Rate": LR,
        "Batch Size": Batch_size,
        "Memory Size": Buffer_size,
        "Discount Rate": Discount_rate,
        "Epsilon Decay": Epsilon_DecayRate,
        "Hidden Layers": 2,  # Hardcoded, Could be changed.
        "Neurons in Hidden Layers": Hidden_dim,
        "Gradient Update": Gradient_update
    }

    with open(os.path.join(agent_dir, "parameters.txt"), 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")





    #################
    # Training loop #
    #################

    for i in range(Episodes):
        # initialize episode
        GameReward = 0
        episode_step = 0
        episode_loss = 0
        episode_gradient_step = 0
        done = False
        start_state = env.reset() # get start state
        state = state_to_input3(start_state).to(device) # convert to input

        # Epsilon decay #
        # Maybe change the way epsilon is decayed

        ## alt Epsilon = max(Epsilon_min, Epsilon - Epsilon_DecayRate)

        Epsilon = max(Epsilon_min, Epsilon*Epsilon_DecayRate)


        ###############
        # Take action #
        ###############

        while (not done) and (episode_step < max_steps):
            # Choose action #
            if np.random.rand() < Epsilon:
                action = np.random.choice(actions)
            else:
                action = torch.argmax(NN(state).detach()).item()
            
            # Get new state and reward #
            new_state, reward, done, score = env.step(action_names[action])
            new_state = state_to_input3(new_state).to(device)
            GameReward += reward
            
            

            ## Save buffers ##

            # Inspiration from Mikkel's code to crewate multiple buffers
            buffer_index = step_count % Buffer_size
            obs_buffer[buffer_index] = state
            obs_next_buffer[buffer_index] = new_state
            action_buffer[buffer_index] = action
            reward_buffer[buffer_index] = reward
            done_buffer[buffer_index] = done
            
            state = new_state


            ###################
            # Gradient update #
            ###################

            if step_count > Batch_size and step_count%Gradient_update==0:
                batch_idx = np.random.choice(np.minimum(
                    Buffer_size, step_count), size=Batch_size, replace=False)
                
                out = NN(obs_buffer[batch_idx])
                val = out[np.arange(Batch_size), action_buffer[batch_idx]]
                with torch.no_grad():
                    out_next = NN(obs_next_buffer[batch_idx])
                    target = reward_buffer[batch_idx] + \
                        Discount_rate*torch.max(out_next, dim=1).values * \
                        (1-done_buffer[batch_idx])
                loss = loss_function(val, target)
                
                # optimizing #
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Data update #
                episode_gradient_step += 1
                episode_loss += loss.item()
                
            episode_step += 1 # Count steps in current episode
            step_count += 1 # Count total steps
            Current_score = score
        GameRewards.append(GameReward)
        # inspired by Mikkel to make loss data this way
        losses.append(episode_loss / (episode_gradient_step+1)) # Average loss per gradient step
        episode_steps.append(episode_step)
        



        ##################
        # Save data per episode to CSV #
        ##################
        with open(os.path.join(agent_dir, "training_data.csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            if i == 0:  # Write header
                writer.writerow(['Episode', 'Steps', 'Steps per Episode','Epsilon', 'Loss', 'GameReward', 'Score'])
            writer.writerow([i + 1, step_count, episode_step, Epsilon, episode_loss, GameReward, Current_score])





        ##################
        # Print progress #
        ##################
        
        # Print progress
        if (i+1) % print_interval == 0:
            average_score = np.mean(GameRewards[-print_interval:-1])
            average_episode_steps = np.mean(episode_steps[-print_interval:-1])
            print(f'Episode={i+1}, Average reward={average_score:.1f}, Steps={average_episode_steps:.0f}, Epsilon={Epsilon:.4f}')
            print(f'Neurons in hidden layer: {Hidden_dim}, Current network: {Current_Network}\n')


    # close environment #
    env.close()

    # training time #
    training_time = time.time() - start_time
    print(f'Training time: {training_time} seconds')
    print('##############################################\n')

    # Save training time in the parameters file #
    with open(os.path.join(agent_dir, "parameters.txt"), 'a') as f:
        f.write(f"\nTraining Time: {training_time} seconds")

    # Save model
    torch.save(NN.state_dict(), os.path.join(agent_dir, 'balls_q_net.pt'))




NN1 = torch.nn.Sequential( 
    torch.nn.Linear(State_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Action_dim)
)

NN2 = torch.nn.Sequential( 
    torch.nn.Linear(State_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Action_dim)
)

NN3 = torch.nn.Sequential( 
    torch.nn.Linear(State_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden_dim, Action_dim)
)

## Run training ##
if __name__ == "__main__":
    train()