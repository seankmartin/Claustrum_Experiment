# Multiple files for behavioural testing and training.

## File types

A pmpc file is a custom file format with slight differences from the standard MPC language.
This should be easier to read than an MPC file.
The script parse_mpc.py provided in two directories up can be used to convert a pmpc file to valid mpc.
The .MPC files should be used for the actual MedPC box.

## Building process

Copy the .MPC files to the default MPC folder for Trans IV. Open Translate and Compile in TRANS IV. Set the .MPC files to Make (M). Hit OK.

## Running

All of the procedures lock the fan on. This must be turned off manually from MPC by turning off locked output 16.
