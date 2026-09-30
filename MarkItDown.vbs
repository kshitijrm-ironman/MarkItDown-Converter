' MarkItDown - silent launcher.
'
' Point your desktop / Start Menu shortcut at THIS file (icon: assets\markitdown.ico).
' It runs "Run MarkItDown.bat" with no console window at all, so launching the app
' just opens the browser.
'
' Exception: if the virtualenv has not been built yet (first run, or after a
' dependency bump) the setup output is worth seeing, so the window is shown.

Option Explicit

Dim sh, fso, base, style

Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

base = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = base

If fso.FileExists(base & "\venv\.deps_installed") Then
    style = 0   ' hidden - everything is installed, the bat exits in a second anyway
Else
    style = 1   ' visible - first-time setup, show pip progress and any errors
End If

sh.Run """" & base & "\Run MarkItDown.bat""", style, False
