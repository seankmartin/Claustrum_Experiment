'''Declare global variables'''

' Number of trials
dim num_trials
dim num_correct
dim elapsed_trials

'timing delays
dim match_delay
dim trial_delay
dim wrong_delay
dim food_delay

'input pins
dim left_lever_inpin
dim right_lever_inpin
dim back_lever_inpin

'output pins
dim left_lever_outpin
dim right_lever_outpin
dim back_lever_outpin
dim left_light_outpin
dim right_light_outpin
dim back_light_outpin
dim house_light_outpin
dim food_outpin

'store input values
dim left_lever_value
dim right_lever_value
dim back_lever_value

'store state
dim left_lever_active
dim right_lever_active
dim back_lever_active
dim left_completed
dim right_completed
dim experiment_state

'store a string to print
dim print_value

'timing
dim start_time
dim elapsed_time
dim back_time

'''Assign variables'''

'setup variables to pins
sub setup_pins()
    left_lever_inpin = 1
    right_lever_inpin = 2
    back_lever_inpin = 3 'TODO find actual value

    left_lever_outpin = 1
    right_lever_outpin = 2
    back_lever_outpin = 3 'TODO find actual value

    left_light_outpin = 9
    right_light_outpin = 10
    back_lever_outpin = 5 'TODO find actual value
    house_light_outpin = 4 'TODO find actual value

    food_outpin = 16
end sub

'''Sub Functions'''

sub init_vars()
    elapsed_trials = 0
    num_correct = 0
    print_value = "Empty"
    start_time = 0
    elapsed_time = 0
    back_time = 0
end sub

sub reset()
    ' Reset all digital outputs to 0
    SignalOut(all) = 0
    left_lever_active = 0
    right_lever_active = 0
    back_lever_active = 0
end sub

sub reset_left_right()
    set_right_side(off)
    set_left_side(off)
end sub

sub set_left_side(state)
    SignalOut(left_light_outpin) = state
    SignalOut(left_lever_outpin) = state

    left_lever_active = state
end sub

sub set_right_side(state)
    SignalOut(right_lever_outpin) = state
    SignalOut(right_light_outpin) = state

    right_lever_active = state
end sub

sub set_back_side(state)
    SignalOut(back_lever_outpin) = state
    SignalOut(back_light_outpin) = state

    back_lever_active = state
end sub

' TODO print value is not necessary if file 1 is accessible outside of main
sub init_experiment()
    ' Do the initial setup
    if (Rnd(1) < 0.5) then
        set_left_side(on)
        print_value = "Trial started with left lever set"
    else
        set_right_side(on)
        print_value = "Trial started with right lever set"
    end if

    left_completed = 0
    right_completed = 0
    start_time = TrialTime
    experiment_state = "Start"
end sub

sub drop_food()
    ' Drop a food pellet
    SignalOut(food_outpin) = on
    DelayMS(15)
    SignalOut(food_outpin) = off
end sub

function generate_back_delay(max)
    ' Generate a random number in the range of 0 to max
    generate_back_delay = trunc(Rnd(1) * (max + 1))

sub show_back_lever()
    dim delay
    delay = generate_back_delay(match_delay)
    DelayMS(delay * 1000)
    set_back_side(on)
    back_time = TrialTime
    experiment_state = "Match"
end sub

sub incorrect_response()
    set_right_side(off)
    set_left_side(off)
    SignalOut(house_light_outpin) = off
    DelayMS(wrong_delay)
    SignalOut(house_light_outpin) = on
    DelayMS(wrong_delay)
end sub

sub correct_response()
    num_correct = num_correct + 1
    set_right_side(off)
    set_left_side(off)
    DelayMS(food_delay)
    drop_food()
    DelayMS(trial_delay - food_delay)
end sub

sub new_experiment()
    elapsed_trials = elapsed_trials + 1
    if (elapsed_trials <= num_trials) then
        init_experiment()
    end if
end sub

sub main()
    ' Run the experiments and record the data
    num_trials = 60 'How long to run the experiment for in seconds
    match_delay = 30 'Max time before the back lever comes out
    trial_delay = 10 'How long between trials in seconds
    wrong_delay = 5 'How long to time out on incorrect response 
    'The wrong delay is usually the trial delay/2
    food_delay = 200 'A small delay before dropping the food in ms

    'convert delays to ms
    trial_delay = trial_delay * 1000
    wrong_delay = wrong_delay * 1000

    Randomize 'Seed the rng based on the system clock

    'this should output in the same name format as other axona files
    open "data.log" for output as #1
    StartUnitRecording

    setup_pins()
    reset()
    init_vars()
    init_experiment() ' TODO might need fixed ratio between random samples
    print #1, print_value
    SignalOut(house_light_outpin) = on

    ' Loop the recording for the desired period of time
    while (elapsed_trials <= num_trials)

        ' Wait for the primate to start by hitting a lever
        if (experiment_state = "Start") then

            if (left_lever_active = 1) then 
                left_lever_value = SignalIn(left_lever_inpin)
                if (left_lever_value = off) 'lever input signal is inverted
                    elapsed_time = TrialTime - start_time
                    print #1, "Pressed left start lever after;", elapsed_time
                    set_left_side(off)
                    left_completed = 1
                    show_back_lever() ' Delay the back lever protrusion
                end if
            end if
            
            if (right_lever_active = 1) then
                right_lever_value = SignalIn(right_lever_inpin)
                if (right_lever_value = off) 'lever input signal is inverted
                    elapsed_time = TrialTime - start_time
                    print #1, "Pressed right start lever after;", elapsed_time
                    set_right_side(off)
                    right_completed = 1
                    show_back_lever() ' Delay the back lever protrusion
                end if
            end if
        end if

        if (experiment_state = "Match") 'TODO could make else if being clever
            if (back_lever_active = 1) then
                back_lever_value = SignalIn(back_lever_inpin)
                if (back_lever_value = off) 'lever input signal is inverted
                    elapsed_time = TrialTime - back_time
                    print #1, "Pressed back lever after;", elapsed_time
                    set_back_side(off)
                    set_left_side(on)
                    set_right_side(on)
                    back_time = TrialTime
                end if

            else 'The back lever has been pressed
                right_lever_value = SignalIn(right_lever_inpin)
                left_lever_value = SignalIn(left_lever_inpin)

                if (right_lever_value = off) 'lever input signal is inverted
                    elapsed_time = TrialTime - back_time
                    reset_left_right()
                    if (right_completed = 1) 'Incorrect response
                        print #1, "Incorrect response after;", elapsed_time
                        incorrect_response()
                    else 'Correct response
                        print #1, "Correct response after;", elapsed_time
                        correct_response()
                    end if
                    new_experiment()

                if (left_lever_value = off) 'lever input signal is inverted
                    elapsed_time = TrialTime - back_time
                    reset_left_right()
                    if (left_completed = 1) 'Incorrect response
                        print #1, "Incorrect response after;", elapsed_time
                        incorrect_response()
                    else 'Correct response
                        print #1, "Correct response after;", elapsed_time
                        correct_response()
                    end if
                    new_experiment()
                end if
            end if
        end if

        ' Don't hog the CPU
        DelayMS(10)
    wend
    StopUnitRecording

    'Print the number of correct responses
    print #1, "Number of trials;", elapsed_trials, ";Number correct;", num_correct

    close #1
end sub