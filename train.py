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
        dolphin_path=None,
        melee_iso_path=None,
        player_stats=["human", "ai"],
        render=True,
        speed=1,
        fullscreen=False,
        audio=False
    )
