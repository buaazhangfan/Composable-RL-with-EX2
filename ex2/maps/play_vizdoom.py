import vizdoom as vzd
# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path, seed):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_labels_buffer_enabled(True)
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    # game.set_mode(vzd.Mode.PLAYER)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    # TO-DO: set seed
    game.set_seed(seed)
    game.init()
    print("Doom initialized.")
    return game

# game = DoomGame()
seed = 300
print('random seed = %d' % seed)
#CONFIGURATION
DEFAULT_CONFIG = '/Users/wangyujie/Desktop/iProud/iCourse/US/294-Reinforcement_Learning/Group_Project/DirectFuturePrediction/maps/D2_navigation.cfg'
game = initialize_vizdoom(DEFAULT_CONFIG, seed)

move = [0.0, 0.0, 1.0]

episodes = 10
for i in range(episodes):
    print("Episode #" +str(i+1))

    game.new_episode()
    while not game.is_episode_finished():
        s = game.get_state()
        # print('s',s)
        game.advance_action()
        a = game.get_last_action()
        print('a', a)
        if a[2] == 1.0:
            print('move')
        r = game.get_last_reward()
        v = game.get_game_variable(vzd.GameVariable.HEALTH)
        # print('v ',v)
        
    t_r = game.get_total_reward()
    print('r',t_r)
