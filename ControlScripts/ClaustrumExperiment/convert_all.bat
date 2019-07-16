@ECHO OFF

set ROOT=C:\Users\smartin5\Repos\neuro-tools\ControlScripts\
set LOCA=%ROOT%\ClaustrumExperiment\MEDPC
set SCRIPTLOC=%ROOT%\parse_mpc.py

echo "Working in %ROOT%"
for %%f in (%LOCA%\*.pmpc) do (
    echo "Parsing %%~nf"
    python3 %SCRIPTLOC% --loc %%f
)