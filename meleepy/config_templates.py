DOLPHIN_INI = """[General]
LastFilename = SSBM.iso
ShowLag = False
ShowFrameCount = False
ISOPaths = 2
RecursiveISOPaths = False
NANDRootPath =
WirelessMac =
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


GALE01_INI = """[Gecko]
{match_setup}

[Gecko_Enabled]
$DMA Read Before Poll
$Skip Memcard Prompt
{speed_hack}
$Boot To Match
$Match Setup
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
"""
#Triggers/L-Analog = `Axis L -+`
#Triggers/R-Analog = `Axis R -+`


BOOT_TO_MATCH = """
$Match Setup
C21B148C 00000025 #BootToMatch.asm
3C608048 60630530
48000021 7C8802A6
38A000F0 3D808000
618C31F4 7D8903A6
4E800421 480000F8
4E800021 0808024C
00000000 000000FF
000000{stage} 00000000
00000000 00000000
00000000 FFFFFFFF
FFFFFFFF 00000000
3F800000 3F800000
3F800000 00000000
00000000 00000000
00000000 00000000
00000000 00000000
00000000 00000000
00000000 {char1}{player1}0400
00FF0000 09007800
400004{cpu1} 00000000
00000000 3F800000
3F800000 3F800000
{char2}{player2}0400 00FF0000
09007800 400004{cpu2}
00000000 00000000
3F800000 3F800000
3F800000 09030400
00FF0000 09007800
40000401 00000000
00000000 3F800000
3F800000 3F800000
09030400 00FF0000
09007800 40000401
00000000 00000000
3F800000 3F800000
3F800000 BB610014
60000000 00000000
"""
