from pathlib import Path


DOLPHIN_INI = """[General]
LastFilename = SSBM.iso
ShowLag = False
ShowFrameCount = False
ISOPaths = 2
RecursiveISOPaths = False
NANDRootPath =
WirelessMac =
EnableSlippiNetworkingOutput = True
[Interface]
ConfirmStop = False
UsePanicHandlers = True
OnScreenDisplayMessages = True
HideCursor = False
AutoHideCursor = False
MainWindowPosX = 50
MainWindowPosY = 50
MainWindowWidth = 400
MainWindowHeight = 328
Language = 0
ShowToolbar = True
ShowStatusbar = True
ShowLogWindow = False
ShowLogConfigWindow = False
ExtendedFPSInfo = False
ThemeName40 = Clean
PauseOnFocusLost = False
[Display]
FullscreenResolution = Auto
Fullscreen = {fullscreen}
RenderToMain = False
RenderWindowXPos = 0
RenderWindowYPos = 0
RenderWindowWidth = 320
RenderWindowHeight = 264
RenderWindowAutoSize = False
KeepWindowOnTop = False
ProgressiveScan = False
PAL60 = True
DisableScreenSaver = True
ForceNTSCJ = False
[GameList]
ListDrives = False
ListWad = True
ListElfDol = True
ListWii = True
ListGC = True
ListJap = True
ListPal = True
ListUsa = True
ListAustralia = True
ListFrance = True
ListGermany = True
ListItaly = True
ListKorea = True
ListNetherlands = True
ListRussia = True
ListSpain = True
ListTaiwan = True
ListWorld = True
ListUnknown = True
ListSort = 3
ListSortSecondary = 0
ColorCompressed = True
ColumnPlatform = True
ColumnBanner = True
ColumnNotes = True
ColumnFileName = False
ColumnID = False
ColumnRegion = True
ColumnSize = True
ColumnState = True
[Core]
HLE_BS2 = True
TimingVariance = 40
CPUCore = 1
Fastmem = True
CPUThread = True
DSPHLE = True
SkipIdle = True
SyncOnSkipIdle = True
SyncGPU = False
SyncGpuMaxDistance = 200000
SyncGpuMinDistance = -200000
SyncGpuOverclock = 1.00000000
FPRF = False
AccurateNaNs = False
DefaultISO =
DVDRoot =
Apploader =
EnableCheats = True
SelectedLanguage = 0
OverrideGCLang = False
DPL2Decoder = False
Latency = 2
MemcardAPath = {user}/GC/MemoryCardA.USA.raw
MemcardBPath = {user}/GC/MemoryCardB.USA.raw
AgpCartAPath =
AgpCartBPath =
SlotA = 255
SlotB = 10
SerialPort1 = 255
BBA_MAC =
SIDevice0 = {port1}
AdapterRumble0 = False
SimulateKonga0 = False
SIDevice1 = {port2}
AdapterRumble1 = False
SimulateKonga1 = False
SIDevice2 = 0
AdapterRumble2 = False
SimulateKonga2 = False
SIDevice3 = 0
AdapterRumble3 = False
SimulateKonga3 = False
WiiSDCard = False
WiiKeyboard = False
WiimoteContinuousScanning = False
WiimoteEnableSpeaker = False
RunCompareServer = False
RunCompareClient = False
EmulationSpeed = {speed}
FrameSkip = 0x00000000
Overclock = 0.75000000
OverclockEnable = False
GFXBackend = {gfx}
GPUDeterminismMode = auto
PerfMapDir =
[Movie]
PauseMovie = False
Author =
DumpFrames = False
DumpFramesSilent = True
ShowInputDisplay = True
[DSP]
EnableJIT = True
DumpAudio = False
DumpUCode = False
Backend = {audio}
Volume = 50
CaptureLog = False
[Input]
BackgroundInput = True
[FifoPlayer]
LoopReplay = True
[NetPlay]
Nickname = Player
ConnectPort = 2626
HostPort = 2626
ListenPort = 0
[Analytics]
Enabled = False
PermissionAsked = True
"""


PIPE_CONFIG = """Buttons/A = `Button A`
Buttons/B = `Button B`
Buttons/X = `Button X`
Buttons/Y = `Button Y`
Buttons/Z = `Button Z`
Main Stick/Up = `Axis MAIN Y +`
Main Stick/Down = `Axis MAIN Y -`
Main Stick/Left = `Axis MAIN X -`
Main Stick/Right = `Axis MAIN X +`
Triggers/L = `Button L`
Triggers/R = `Button R`
D-Pad/Up = `Button D_UP`
D-Pad/Down = `Button D_DOWN`
D-Pad/Left = `Button D_LEFT`
D-Pad/Right = `Button D_RIGHT`
Buttons/Start = `Button START`
C-Stick/Up = `Axis C Y +`
C-Stick/Down = `Axis C Y -`
C-Stick/Left = `Axis C X -`
C-Stick/Right = `Axis C X +`
Triggers/L-Analog = `Axis L -+`
Triggers/R-Analog = `Axis R -+`
"""


def create_directory(path):
    if not path.is_dir():
        path.mkdir()


def setup_pipe_config(config_directory, player_stats):
    with open(config_directory.joinpath("GCPadNew.ini"), "w") as f:
        config = ""

        for player_id, player_stat in enumerate(player_stats):
            if player_stat == "ai":
                player_number = player_id + 1
                config += "[GCPad%d]\n" % player_number
                config += "Device = Pipe/p%d\n" % player_number
                config += PIPE_CONFIG

        f.write(config)


def setup_dolphin_config(config_directory, user_directory, render, audio, speed, fullscreen, player_stats):
    with open(config_directory.joinpath("Dolphin.ini"), "w") as f:
        f.write(DOLPHIN_INI.format(
            user=user_directory,
            gfx="OGL" if render else "Null",
            audio="Pulse" if audio else "No audio backend",
            speed=speed,
            fullscreen=fullscreen,
            port1 = 12 if player_stats[0] == "human" else 6,
            port2 = 12 if player_stats[1] == "human" else 6,
        ))


def setup_dolphin_user(in_directory=None,
                       player_stats=["human", "ai"],
                       render=True,
                       speed=0,
                       fullscreen=False,
                       audio=False):

    in_directory = Path(in_directory)

    user_directory = in_directory.joinpath("DolphinUser")
    config_directory = user_directory.joinpath("Config")

    create_directory(in_directory)
    create_directory(user_directory)
    create_directory(config_directory)

    setup_pipe_config(config_directory, player_stats)
    setup_dolphin_config(config_directory, user_directory, render, audio, speed, fullscreen, player_stats)

    return user_directory
