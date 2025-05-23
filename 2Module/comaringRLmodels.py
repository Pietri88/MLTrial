import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

grid_size = 5
n_actions = 4
n_states = grid_size * grid_size

Q_table = np.zeros((n_states, n_actions))

alpha = 0.1  # Współczynnik uczenia się
gamma = 0.9  # Współczynnik dyskontowania przyszłych nagród
epsilon = 0.1  # Wskaźnik eksploracji dla polityki epsilon-zachłannej

rewards = np.full((n_states,), -1)
rewards[24] = 10  # Nagroda za osiągnięcie celu
rewards[12] = -10  # Kara za błąd

def epsilon_greedy_action(Q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)  # Eksploracja: losowa akcja
    else:
        return np.argmax(Q_table[state])  # Eksploatacja: akcja o najwyższej wartości Q

# Trening Q-learningu
q_learning_rewards = []
for episode in range(1000):
    state = np.random.randint(0, n_states)  # Start w losowym stanie
    done = False
    total_reward = 0
    
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)  # Symulowany następny stan
        reward = rewards[next_state]
        total_reward += reward
        
        # Aktualizacja Q-wartości
        Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])
        
        state = next_state
        if next_state in {24, 12}:
            done = True
    
    q_learning_rewards.append(total_reward)

# Model do policy gradients
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    tf.keras.layers.Dense(n_actions, activation='softmax')  # Wyjściowe prawdopodobieństwa akcji
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def get_action(state):
    state_input = tf.one_hot(state, n_states)
    action_probs = model(state_input[np.newaxis, :])
    return np.random.choice(n_actions, p=action_probs.numpy()[0])

def compute_cumulative_rewards(rewards, gamma=0.99):
    cumulative_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        cumulative_rewards[t] = running_add
    return cumulative_rewards

def update_policy(states, actions, rewards):
    cumulative_rewards = compute_cumulative_rewards(rewards)
    with tf.GradientTape() as tape:
        state_inputs = tf.one_hot(states, n_states)
        action_probs = model(state_inputs)
        action_masks = tf.one_hot(actions, n_actions)
        log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs), axis=1)
        loss = -tf.reduce_mean(log_probs * cumulative_rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Trening policy gradients
policy_gradients_rewards = []
for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False
    states = []
    actions = []
    episode_rewards = []
    total_reward = 0
    
    while not done:
        action = get_action(state)
        next_state = np.random.randint(0, n_states)
        reward = rewards[next_state]
        total_reward += reward
        
        states.append(state)
        actions.append(action)
        episode_rewards.append(reward)
        
        state = next_state
        if next_state in {24, 12}:
            done = True
    
    update_policy(states, actions, episode_rewards)
    policy_gradients_rewards.append(total_reward)

# Wykres porównujący metody
plt.plot(np.cumsum(q_learning_rewards), label='Q-Learning')
plt.plot(np.cumsum(policy_gradients_rewards), label='Policy Gradients')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Rewards')
plt.legend()
plt.show()
