#reinforcement learner
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def nueral_network(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
                   activation='linear', loss='mse', learning_rate=1.0):
    '''
    n_obs: state size, number of
    n_action: action size, buy, sell, hold
    '''
    model = Sequential()
    print("model state", n_obs)
    model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
    for _ in range(n_hidden_layer):
        model.add(Dense(n_neuron_per_layer, activation=activation))
    model.add(Dense(n_action, activation=activation))
    model.compile(loss=loss, optimizer=Adam())
    print(model.summary())

    return model