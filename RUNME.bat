@echo off
@echo Simulation Starting...
REM Define variables 
set n=50
set R0=3.0856776e10
set timespan=1e6
set theta=0.5
set bound=10

REM Run the Python script with the variables
python Lagman_Darian_nBody.py --n %n% --R0 %R0% --timespan %timespan% --theta %theta% --bound-condition %bound%
pause