#define AppName "隐变智模"
#define AppVersion "1.0.0"
#define AppPublisher "Science OpenMyCode"
#define BuildRoot "..\..\dist\YinbianZhimo"
#define LauncherRoot "bin"

[Setup]
AppId={{A67D9160-CDE7-4518-9693-35B2CE84A25A}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\YinbianZhimo\versions\{#AppVersion}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename=YinbianZhimo-{#AppVersion}-Windows-x64-Offline
Compression=lzma2/fast
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName} {#AppVersion}
SetupLogging=yes

[Tasks]
Name: desktopicon; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式"
Name: addtopath; Description: "将 yinbian CLI 加入当前用户 PATH"; GroupDescription: "命令行"

[Files]
Source: "{#BuildRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "vendor\WebView2\*"; DestDir: "{app}\WebView2"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#LauncherRoot}\YinbianLauncherGui.exe"; DestDir: "{localappdata}\Programs\YinbianZhimo\launcher"; DestName: "隐变智模部署.exe"; Flags: ignoreversion
Source: "{#LauncherRoot}\YinbianLauncherGui.exe"; DestDir: "{localappdata}\Programs\YinbianZhimo\launcher"; DestName: "隐变智模对话.exe"; Flags: ignoreversion
Source: "{#LauncherRoot}\YinbianLauncherConsole.exe"; DestDir: "{localappdata}\Programs\YinbianZhimo\launcher"; DestName: "yinbian.exe"; Flags: ignoreversion

[Icons]
Name: "{group}\隐变智模部署"; Filename: "{localappdata}\Programs\YinbianZhimo\launcher\隐变智模部署.exe"
Name: "{group}\隐变智模对话"; Filename: "{localappdata}\Programs\YinbianZhimo\launcher\隐变智模对话.exe"
Name: "{autodesktop}\隐变智模部署"; Filename: "{localappdata}\Programs\YinbianZhimo\launcher\隐变智模部署.exe"; Tasks: desktopicon
Name: "{autodesktop}\隐变智模对话"; Filename: "{localappdata}\Programs\YinbianZhimo\launcher\隐变智模对话.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{localappdata}\Programs\YinbianZhimo\launcher"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{localappdata}\Programs\YinbianZhimo\launcher'))

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\YinbianZhimo\cache\models"; Check: DeleteModels
Type: filesandordirs; Name: "{localappdata}\YinbianZhimo\credentials"; Check: DeleteKeys
Type: filesandordirs; Name: "{localappdata}\YinbianZhimo\state"; Check: DeleteServers
Type: filesandordirs; Name: "{localappdata}\YinbianZhimo\chat"; Check: DeleteChat
Type: filesandordirs; Name: "{localappdata}\YinbianZhimo\logs"; Check: DeleteLogs

[Code]
var
  KeepModels, KeepKeys, KeepServers, KeepChat, KeepLogs: TNewCheckBox;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ProductRoot, CurrentPath, TemporaryPath, PreviousPath, Payload, InstallPath: string;
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then Exit;
  ProductRoot := ExpandConstant('{localappdata}\Programs\YinbianZhimo');
  CurrentPath := ProductRoot + '\current.json';
  TemporaryPath := ProductRoot + '\current.json.partial';
  PreviousPath := ProductRoot + '\current.previous.json';
  ForceDirectories(ProductRoot + '\launcher');
  if FileExists(CurrentPath) then begin
    DeleteFile(PreviousPath);
    CopyFile(CurrentPath, PreviousPath, False);
  end;
  InstallPath := ExpandConstant('{app}');
  if not Exec(InstallPath + '\yinbian.exe', '--help', '', SW_HIDE,
    ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
    RaiseException('新版本自检失败，current.json 保持旧版本');
  StringChangeEx(InstallPath, '\', '\\', True);
  Payload := '{"version":"{#AppVersion}","path":"' +
    InstallPath + '","validated":true}';
  SaveStringToFile(TemporaryPath, Payload, False);
  DeleteFile(CurrentPath);
  if not RenameFile(TemporaryPath, CurrentPath) then
    RaiseException('无法原子更新 current.json');
end;

function NeedsAddPath(Path: string): Boolean;
var CurrentPath: string;
begin
  RegQueryStringValue(HKCU, 'Environment', 'Path', CurrentPath);
  Result := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(CurrentPath) + ';') = 0;
end;

function DeleteModels(): Boolean; begin Result := not KeepModels.Checked; end;
function DeleteKeys(): Boolean; begin Result := not KeepKeys.Checked; end;
function DeleteServers(): Boolean; begin Result := not KeepServers.Checked; end;
function DeleteChat(): Boolean; begin Result := not KeepChat.Checked; end;
function DeleteLogs(): Boolean; begin Result := not KeepLogs.Checked; end;

procedure InitializeUninstallProgressForm;
begin
  KeepModels := TNewCheckBox.Create(UninstallProgressForm);
  KeepModels.Parent := UninstallProgressForm;
  KeepModels.Caption := '保留模型和转换工件（默认）'; KeepModels.Checked := True;
  KeepModels.Left := 20; KeepModels.Top := 150;
  KeepKeys := TNewCheckBox.Create(UninstallProgressForm);
  KeepKeys.Parent := UninstallProgressForm;
  KeepKeys.Caption := '保留密钥（默认）'; KeepKeys.Checked := True;
  KeepKeys.Left := 20; KeepKeys.Top := 175;
  KeepServers := TNewCheckBox.Create(UninstallProgressForm);
  KeepServers.Parent := UninstallProgressForm;
  KeepServers.Caption := '保留服务器和部署记录'; KeepServers.Checked := True;
  KeepServers.Left := 20; KeepServers.Top := 200;
  KeepChat := TNewCheckBox.Create(UninstallProgressForm);
  KeepChat.Parent := UninstallProgressForm;
  KeepChat.Caption := '保留本地会话历史'; KeepChat.Checked := True;
  KeepChat.Left := 20; KeepChat.Top := 225;
  KeepLogs := TNewCheckBox.Create(UninstallProgressForm);
  KeepLogs.Parent := UninstallProgressForm;
  KeepLogs.Caption := '保留日志'; KeepLogs.Checked := True;
  KeepLogs.Left := 20; KeepLogs.Top := 250;
end;
