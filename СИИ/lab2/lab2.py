import numpy as np

class GridWorld:
    def __init__(self, grid, start_pos=(0, 0), max_steps=200):
        self.grid = np.array(grid)
        self.start_pos = start_pos
        self.max_steps = max_steps
        self.current_pos = start_pos
        self.steps_taken = 0
        self.n_rows, self.n_cols = self.grid.shape

    def reset(self):
        """Сбросить среду в начальное состояние."""
        self.current_pos = self.start_pos
        self.steps_taken = 0
        return self.current_pos

    def step(self, action):
        """
        Выполнить действие агента.
        :param action: 0=вверх, 1=вниз, 2=влево, 3=вправо
        :return: next_state, reward, done
        """
        x, y = self.current_pos
        new_x, new_y = x, y

        # Определяем новое положение агента
        if action == 0:  # Вверх
            new_x = max(x - 1, 0)
        elif action == 1:  # Вниз
            new_x = min(x + 1, self.n_rows - 1)
        elif action == 2:  # Влево
            new_y = max(y - 1, 0)
        elif action == 3:  # Вправо
            new_y = min(y + 1, self.n_cols - 1)

        # Проверяем, не стена ли новая позиция
        if self.grid[new_x, new_y] == 1:
            new_x, new_y = x, y  # Возвращаемся на место, если стена

        self.current_pos = (new_x, new_y)
        self.steps_taken += 1

        # Определяем награду и завершение эпизода
        cell_value = self.grid[new_x, new_y]
        if cell_value == 2:  # Золото
            reward, done = 100, True
        elif cell_value == -1:  # Ловушка
            reward, done = -100, True
        elif self.steps_taken >= self.max_steps:  # Превышен лимит шагов
            reward, done = -1, True
        else:  # Обычная клетка
            reward, done = -1, False

        return (new_x, new_y), reward, done
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Скорость обучения
        self.gamma = gamma  # Коэффициент дисконтирования
        self.epsilon = epsilon  # Вероятность исследования
        self.Q = {}  # Таблица Q(s,a)

    def get_q_value(self, state, action=None):
        """Получить Q-значение для состояния и действия."""
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        if action is not None:
            return self.Q[state][action]
        return self.Q[state]

    def choose_action(self, state):
        """Выбрать действие с использованием ε-жадной стратегии."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)  # Случайное действие
        else:
            return np.argmax(self.get_q_value(state))  # Жадное действие

    def update_q(self, state, action, reward, next_state, done):
        """Обновить Q-значение по формуле Q-learning."""
        current_q = self.get_q_value(state, action)
        max_next_q = np.max(self.get_q_value(next_state)) if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state][action] = new_q
def train_agent(env, agent, episodes=2000):
    episode_returns = []
    episode_lengths = []
    success_rates = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        success_rates.append(1 if total_reward == 100 else 0)

    return episode_returns, episode_lengths, success_rates
grid = [
    [0, 0, 0, 0, -1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 2, 0]
]

env = GridWorld(grid, start_pos=(0, 0))
agent = QLearningAgent(n_states=25, n_actions=4)  # 25 состояний (5x5), 4 действия
episode_returns, episode_lengths, success_rates = train_agent(env, agent, episodes=2000)
import matplotlib.pyplot as plt

def plot_results(episode_returns, episode_lengths, success_rates):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(episode_returns)
    plt.title("Суммарная награда за эпизод")
    plt.xlabel("Эпизод")
    plt.ylabel("Награда")

    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths)
    plt.title("Длина эпизода")
    plt.xlabel("Эпизод")
    plt.ylabel("Шаги")

    plt.subplot(1, 3, 3)
    plt.plot(np.convolve(success_rates, np.ones(50)/50, mode='valid'))
    plt.title("Скользящая доля успешных эпизодов")
    plt.xlabel("Эпизод")
    plt.ylabel("Доля успехов")

    plt.tight_layout()
    plt.show()

plot_results(episode_returns, episode_lengths, success_rates)
