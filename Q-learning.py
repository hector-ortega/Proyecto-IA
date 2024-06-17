import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Dividimos la posicion y la velocidad en segmentos
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # entre -1.2 y 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # entre -0.07 y 0.07

    if(is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        f = open('mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # tasa de aprendizaje Controla cuán rápido o lento el agente ajusta su conocimiento sobre el entorno. Un valor alto de 𝛼α significa que el agente da más peso a la nueva información, mientras que un valor bajo hace que el agente aprenda más lentamente.
    #    Controla cuánto valora el agente las recompensas a largo plazo en comparación con las recompensas inmediatas. Un 𝛾
    #  γ cercano a 1 hace que el agente se preocupe más por las recompensas futuras, mientras que un  𝛾 
    # γ cercano a 0 hace que se concentre en las recompensas inmediatas.
    discount_factor_g = 0.9 

    epsilon = 1  #El parámetro de exploración, comúnmente denotado como 𝜖 ϵ, determina la probabilidad de que el agente explore el entorno eligiendo acciones aleatorias en lugar de explotar el conocimiento existente para maximizar la recompensa.
    epsilon_decay_rate = 2/episodes #  controla la velocidad con la que  𝜖 ϵ disminuye, ajustando la probabilidad de exploración a lo largo de los episodios.
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes) #almacena recompenzas acumuladas en el episodio

    for i in range(episodes):
        state = env.reset()[0]      # posicion inicial siempre es 0 al igual que la velocidad
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # Verdadero cuando alcanza el objetivo

        rewards=0

        while(not terminated and rewards>-1000):

            if is_training and rng.random() < epsilon:
                # toma una accion (0=drive left, 1=stay neutral, 2=drive right) y obtiene el nuevo estado y la recompenza
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action] # se actualiza la tabla Q si esta en entrenamiento
                )

            #actualiza el estado actual y acumula las recompenzas
            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards+=reward

        epsilon = max(epsilon - epsilon_decay_rate, 0) #reduce el valor de epsilon

        rewards_per_episode[i] = rewards #guarda la recompenza total del episodio

    env.close()

    # se guarda la tabla Q en un archivo para su futura consulta
    if is_training:
        f = open('mountain_car.pkl','wb')
        pickle.dump(q, f)
        f.close()

    # Se genera un grafico de las recompenzas
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')

if __name__ == '__main__':

    run(5000, is_training=True, render=False)