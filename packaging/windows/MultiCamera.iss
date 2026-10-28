; Inno Setup 6 — 将 PyInstaller 输出的 dist\MultiCamera 打成 Windows 安装程序
; 使用 scripts\build_windows_installer.bat，自动传入唯一来源中的产品版本。

#define MyAppName "MultiCamera Calibration"
#define MyAppShortName "MultiCamera"
#ifndef MyAppVersion
  #error "Pass /DMyAppVersion from scripts/check_release.py --print-version"
#endif
#define MyAppPublisher "MultiCamera"
#define MyAppExeName "MultiCamera.exe"
#ifndef DistDir
  #define DistDir "..\\..\\dist\\MultiCamera"
#endif
#ifndef OutputRoot
  #define OutputRoot "..\\..\\installer_output"
#endif
#ifndef MyCompression
  #define MyCompression "lzma2/normal"
#endif

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
Compression={#MyCompression}
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\..\assets\logo.ico
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
