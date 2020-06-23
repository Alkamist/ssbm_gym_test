from meleepy.melee import Melee


#melee_options = dict(
#    render=True,
#    speed=0,
#    player1='ai',
#    player2='human',
#    char1='falcon',
#    char2='falcon',
#    stage='final_destination',
#)


if __name__ == "__main__":
    melee = Melee(
        dolphin_user_directory="C:\\Users\\Corey\\Documents\\Dolphin Emulator",
        player_stats=["human", "human"],
        render=True,
        speed=1,
        fullscreen=False,
        audio=False,
    )

    melee._start_process()

#    melee.reset()

#    while True:
#        state = melee.step()
#
#        #print(state.players[1].x)
