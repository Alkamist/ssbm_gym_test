from pathlib import Path


DOLPHIN_INI = """[General]
LastFilename = SSBM.iso
ShowLag = False
ShowFrameCount = False
ISOPaths = 0
RecursiveISOPaths = False
NANDRootPath =
DumpPath =
WirelessMac =
WiiSDCardPath = {user}/Wii/sd.raw
SlippiConsoleName = Dolphin
EnableSlippiNetworkingOutput = False
[Interface]
ConfirmStop = False
UsePanicHandlers = True
OnScreenDisplayMessages = True
HideCursor = False
AutoHideCursor = False
MainWindowPosX = 300
MainWindowPosY = 300
MainWindowWidth = 400
MainWindowHeight = 328
LanguageCode =
ShowToolbar = True
ShowStatusbar = True
ShowLogWindow = False
ShowLogConfigWindow = False
ExtendedFPSInfo = False
ThemeName = Clean
PauseOnFocusLost = False
DisableTooltips = False
[Display]
FullscreenResolution = Auto
Fullscreen = {fullscreen}
RenderToMain = False
RenderWindowXPos = 200
RenderWindowYPos = 200
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
HLE_BS2 = False
TimingVariance = 40
CPUCore = 1
Fastmem = True
CPUThread = True
DSPHLE = True
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
TimeStretching = False
RSHACK = False
Latency = 2
MemcardAPath = {user}/GC/MemoryCardA.USA.raw
MemcardBPath = {user}/GC/MemoryCardB.USA.raw
AgpCartAPath =
AgpCartBPath =
SlotA = 255
SlotB = 255
SerialPort1 = 255
BBA_MAC =
SIDevice0 = {port1}
AdapterRumble0 = True
SimulateKonga0 = False
SIDevice1 = {port2}
AdapterRumble1 = True
SimulateKonga1 = False
SIDevice2 = 0
AdapterRumble2 = True
SimulateKonga2 = False
SIDevice3 = 0
AdapterRumble3 = True
SimulateKonga3 = False
WiiSDCard = False
WiiKeyboard = False
WiimoteContinuousScanning = False
WiimoteEnableSpeaker = False
RunCompareServer = False
RunCompareClient = False
EmulationSpeed = {speed}
FrameSkip = 0x00000000
Overclock = 1.00000000
OverclockEnable = False
GFXBackend = {gfx}
GPUDeterminismMode = auto
PerfMapDir =
EnableCustomRTC = False
CustomRTCValue = 0x386d4380
AllowAllNetplayVersions = False
QoSEnabled = True
AdapterWarning = True
ShownLagReductionWarning = False
[Movie]
PauseMovie = False
Author =
DumpFrames = False
DumpFramesSilent = False
ShowInputDisplay = False
ShowRTC = False
[DSP]
EnableJIT = True
DumpAudio = False
DumpAudioSilent = False
DumpUCode = False
Backend = {audio}
Volume = 100
CaptureLog = False
[Input]
BackgroundInput = True
[FifoPlayer]
LoopReplay = True
[Analytics]
ID = 0
Enabled = False
PermissionAsked = True
[Network]
SSLDumpRead = False
SSLDumpWrite = False
SSLVerifyCert = False
SSLDumpRootCA = False
SSLDumpPeerCert = False
[BluetoothPassthrough]
Enabled = False
VID = -1
PID = -1
LinkKeys =
[Sysconf]
SensorBarPosition = 1
SensorBarSensitivity = 50331648
SpeakerVolume = 88
WiimoteMotor = True
WiiLanguage = 1
AspectRatio = 1
Screensaver = 0
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


GALE01_INI = """[Gecko_Enabled]
$Faster Melee Netplay Settings
$Normal Lag Reduction
$Slippi Recording
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


def setup_melee_config(game_settings_directory):
    with open(game_settings_directory.joinpath("GALE01.ini"), "w") as f:
        f.write(GALE01_INI)


def setup_dolphin_user(user_directory=None,
                       player_stats=["human", "ai"],
                       render=True,
                       speed=0,
                       fullscreen=False,
                       audio=False):
    config_directory = user_directory.joinpath("Config")
    game_settings_directory = user_directory.joinpath("GameSettings")

    create_directory(user_directory)
    create_directory(config_directory)
    create_directory(game_settings_directory)

    #setup_pipe_config(config_directory, player_stats)
    #setup_dolphin_config(config_directory, user_directory, render, audio, speed, fullscreen, player_stats)
    #setup_melee_config(game_settings_directory)
