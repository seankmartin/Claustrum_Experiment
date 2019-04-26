'''Declare global variables'''

' Experiment time set here
dim experimenttime

'input pins
dim left_lever_inpin
dim right_lever_inpin

'output pins
dim left_lever_outpin
dim right_lever_outpin
dim left_light_outpin
dim right_light_outpin
dim food_outpin

'store input values
dim left_lever_value
dim right_lever_value

'store state
dim left_lever_active
dim right_lever_active
dim left_completed
dim right_completed

'''Assign variables'''

'setup variables to pins
sub setup_pins()
    left_lever_inpin = 1
    right_lever_inpin = 2

    left_lever_outpin = 1
    right_lever_outpin = 2
    left_light_outpin = 9
    right_light_outpin = 10
    food_outpin = 16
end sub

'''Sub Functions'''

sub reset()
    ' Reset all digital outputs to 0
    SignalOut(all) = 0
    left_lever_active = 0
    right_lever_active = 0
end sub

sub set_first_side(state)
    ' Denote that the left lever should shoot out
    SignalOut(left_light_outpin) = state
    SignalOut(left_lever_outpin) = state

    left_lever_active = state
end sub

sub set_second_side(state)
    SignalOut(right_lever_outpin) = state
    SignalOut(right_light_outpin) = state

    right_lever_active = state
end sub

sub init_experiment()
    ' Do the initial setup
    if (Rnd(1) < 0.5) then
        set_first_side(on)
    else
        set_second_side(on)
    end if

    left_completed = 0
    right_completed = 0
end sub

sub drop_food()
    ' Drop a food pellet
    SignalOut(food_outpin) = on
    DelayMS(15)
    SignalOut(food_outpin) = off
end sub

sub main()
    ' Run the experiments and record the data
    experimenttime = 60 'How long to run the experiment for

    Randomize 'Seed the rng based on the system clock

    setup_pins()
    reset()
    init_experiment()

    StartUnitRecording
    while (TrialTime/1000 < experimenttime)

        ' Wait for the primate to start by hitting a lever
        left_lever_value = SignalIn(left_lever_inpin)
        right_lever_value = SignalIn(right_lever_inpin)
        
        if (left_lever_value = off) and (left_lever_active = 1) then
            print "pressed first side"
            set_first_side(off)
            if (right_completed = 0) then
                DelayMS(200)
                set_second_side(on)
            end if
            left_completed = 1
        end if

        if (right_lever_value = off) and (right_lever_active = 1) then
            print "pressed second side"
            set_second_side(off)
            if (left_completed = 0) then
                DelayMS(200)
                set_first_side(on)
            end if
            right_completed = 1
        end if

        if (left_completed = 1) and (right_completed = 1) then
            reset()
            DelayMS(1000) ' A delay before dropping the food
            drop_food()
            DelayMS(1000) ' A delay before restarting the experiment
            init_experiment()
        end if

        ' Don't hog the CPU
        DelayMS(10)
    wend
    StopUnitRecording
end sub