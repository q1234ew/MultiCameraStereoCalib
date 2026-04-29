; Inno Setup 6 — 将 PyInstaller 输出的 dist\MultiCamera 打成 Windows 安装程序
; 编译：在已安装 Inno Setup 6 的 Windows 上执行 ISCC packaging\windows\MultiCamera.iss
; （版本号请与 pyproject.toml [project] version 保持一致）

#define MyAppName "MultiCamera Calibration"
#define MyAppShortName "MultiCamera"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "MultiCamera"
#define MyAppExeName "MultiCamera.exe"
#define DistDir "..\\..\\dist\\MultiCamera"
#define OutputRoot "..\\..\\installer_output"

[Setup]
AppId={{8F3C2A1B-9D0E-4F2A-8C7B-1E2D3A4B5C6D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppShortName}
DefaultGroupName={#MyAppShortName}
AllowNoIcons=yes
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputRoot}
OutputBaseFilename=MultiCamera_Setup_{#MyAppVersion}_x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0
DisableProgramGroupPage=no
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppShortName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppShortName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppShortName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
