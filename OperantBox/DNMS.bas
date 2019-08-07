'''Created by Sean Martin on 26/04/2019 for Matheus Oliveira'''
'''Contact martins7@tcd.ie'''
'''Please note that lever input signals are inverted'''

'''Declare global variables'''

' Number of trials
dim num_trials
dim num_correct
dim elapsed_trials

' Store per trial information
dim trial_sides
dim delay_times

'timing delays
dim max_match_delay
dim trial_delay
dim wrong_delay
dim reward_delay
dim show_front_delay

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
dim experiment_state

'timing
dim start_time
dim elapsed_time
dim back_time

' Optional experimenter tag
dim tag
'''Varable init routines'''

sub setup_pins()
    'setup variables to pins
    left_lever_inpin = 1
    right_lever_inpin = 2
    back_lever_inpin = 3

    left_lever_outpin = 1
    right_lever_outpin = 2
    back_lever_outpin = 3
    left_light_outpin = 9
    right_light_outpin = 10
    back_light_outpin = 11
    house_light_outpin = 12
    food_outpin = 16
end sub

sub init_vars()
    'init variables such as trials and times
    elapsed_trials = 0
    num_correct = 0
    start_time = 0
    elapsed_time = 0
    back_time = 0
    left_lever_active = 0
    right_lever_active = 0
    back_lever_active = 0
end sub

sub init_arrays()
    ' Setup arrays which hold info for each trial
    dim indices
    indices = [0, num_trials-1]
    trial_sides = vararraycreate(indices, 12)
    delay_times = vararraycreate(indices, 12)

    dim half_point
    half_point = trunc(num_trials / 2)

    dim i
    for i = 0 to half_point - 1
        trial_sides[i] = 1 'Left is 1
    next
    for i = half_point to num_trials - 1
        trial_sides[i] = 0 'Right is 0
    next
end sub


'''Signal Output related routines'''

sub reset()
    'Reset all digital outputs to 0
    SignalOut(all) = 0
end sub

sub set_left_side(state)
    ' Set left light and lever to state
    SignalOut(left_light_outpin) = state
    SignalOut(left_lever_outpin) = state

    left_lever_active = state
end sub

sub set_right_side(state)
    ' Set right light and lever to state
    SignalOut(right_lever_outpin) = state
    SignalOut(right_light_outpin) = state

    right_lever_active = state
end sub

sub reset_left_right()
    ' Set right and left sides off
    set_right_side(off)
    set_left_side(off)
end sub

sub set_back_side(state)
    ' Set back light and lever to state
    SignalOut(back_lever_outpin) = state
    SignalOut(back_light_outpin) = state

    back_lever_active = state
end sub

sub deliver_reward()
    ' Drop a food pellet
    SignalOut(food_outpin) = on
    DelayMS(15)
    SignalOut(food_outpin) = off
end sub


'''Maths related subroutines'''

function generate_random_float(max)
    ' Generate a random number in the range of 0 to max
    generate_random_float = trunc(Rnd(1) * (max + 1))
end function

sub knuth_shuffle(byref in_array, array_len)
    'Randomly shuffle an array - see
    'https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    dim i
    dim j
    dim temp
    for i = array_len - 1 to 1 step -1
        j = generate_random_float(i)
        temp = in_array[j]
        in_array[j] = in_array[i]
        in_array[i] = temp
    next
end sub


'''Experiment Control Routines'''

sub new_experiment(first)
    ' Begin a new trial
    dim side
    dim side_nice
    if (first = 0) then
        elapsed_trials = elapsed_trials + 1
    end if

    if (elapsed_trials < num_trials) then
        if (trial_sides[elapsed_trials] = 1) then
            set_left_side(on)
            side = ";Left;"
            side_nice = "left lever"
        else
            set_right_side(on)
            side = ";Right;"
            side_nice = "right lever"
        end if
        print "Starting Trial number ", elapsed_trials+1, " Out of ", num_trials
        print "Showing the subject ", side_nice, ", will delay ", delay_times[elapsed_trials], " seconds before the matching"
        print #1, "Trial;", elapsed_trials+1
        print #1, "Begin;", TrialTime, side, delay_times[elapsed_trials]

        start_time = TrialTime
        experiment_state = "Start"
    end if
end sub

sub incorrect_response()
    ' Turn off the house light before starting next trial
    set_right_side(off)
    set_left_side(off)
    SignalOut(house_light_outpin) = off
    DelayMS(wrong_delay)
    SignalOut(house_light_outpin) = on
    DelayMS(wrong_delay)
end sub

sub correct_response()
    ' Give the subject some reward for being smart
    num_correct = num_correct + 1
    set_right_side(off)
    set_left_side(off)
    DelayMS(reward_delay)
    deliver_reward()
    DelayMS(trial_delay - reward_delay)
end sub

sub show_back_lever()
    ' Delay for a while and then show the back lever
    dim delay
    delay = delay_times[elapsed_trials]
    DelayMS(delay * 1000)
    set_back_side(on)
    back_time = TrialTime
    experiment_state = "Back"
end sub

sub start_lever_pressed(side)
    ' Call after the start lever is pressed, side = 1 denotes left
    elapsed_time = TrialTime - start_time
    print #1, "Front;", TrialTime, ";", elapsed_time

    if (side = 1) then
        set_left_side(off)
    else
        set_right_side(off)
    end if

    show_back_lever() ' Delay the back lever protrusion
end sub

sub back_lever_pressed()
    ' Call after the back lever is pressed
    elapsed_time = TrialTime - back_time
    print #1, "Back;", TrialTime, ";", elapsed_time
    set_back_side(off)
    DelayMS(show_front_delay)
    set_left_side(on)
    set_right_side(on)
    back_time = TrialTime
    experiment_state = "End"
end sub

sub end_experiment(correct)
    elapsed_time = TrialTime - back_time
    reset_left_right()
    if correct then
        correct_response()
    else
        incorrect_response()
    end if

    print "Ending trial, was the subject correct in this trial? ", correct
    print #1, "End;", TrialTime, ";", correct, ";", elapsed_time
    print #1, ";"
    new_experiment(0)
end sub

sub full_init_before_record()
    'Perform script init before recording
    dim i
    'convert some delays to ms
    trial_delay = trial_delay * 1000
    wrong_delay = wrong_delay * 1000
    Randomize 'Seed the rng based on the system clock
    setup_pins()
    reset()
    init_vars()
    init_arrays()
    knuth_shuffle(trial_sides, num_trials)
    SignalOut(house_light_outpin) = on

    ' Generate time delays before showing lever
    for i = 0 to num_trials - 1
        delay_times[i] = 1 + generate_random_float(max_match_delay-1)
    next

    ' Print the csv file header
    print #1, tag, ";", num_trials, ";", max_match_delay, ";", trial_delay, ";", wrong_delay
    print #1, ";"
end sub

''' Main subroutine follows this'''

sub main()
    ' Run the experiments and record the data
    '''NB Change important variables here'''
    num_trials = 60 'Number of trials should be divisible by 2
    max_match_delay = 30 'Max time before the back lever comes out
    trial_delay = 10 'How long between trials in seconds
    wrong_delay = 5 'How long to time out on incorrect response
    'The wrong delay is usually the trial delay/2
    reward_delay = 200 'A small delay before dropping the reward in ms
    show_front_delay = 1000 'Measured in ms
    'A small delay before showing the front levers after the back in ms
    tag = "Optional tag" ' You may tag this experiment if you wish
    'this should output in the same name format as other axona files
    open "data.log" for output as #1

    dim correct
    full_init_before_record()
    StartUnitRecording
    new_experiment(1)

    ' Loop the recording for the number of trials
    while (elapsed_trials < num_trials)

        ' Wait for the subject to start by hitting a lever
        if (experiment_state = "Start") then
            if (left_lever_active = 1) and (SignalIn(left_lever_inpin) = off) then
                start_lever_pressed(1)
            elseif (right_lever_active = 1) and (SignalIn(right_lever_inpin) = off) then
                start_lever_pressed(0)
            end if
        elseif (experiment_state = "Back") then
            if (SignalIn(back_lever_inpin) = off) then
                back_lever_pressed()
            end if
        else 'Checking if the subject can remember the right lever
            if (SignalIn(right_lever_inpin) = off) then
                print "Pressed ", 0, "correct ", 1-trial_sides[elapsed_trials]
                correct = (trial_sides[elapsed_trials] = 1)
                end_experiment(correct)
            elseif (SignalIn(left_lever_inpin) = off) then
                print "Pressed ", 1, "correct ", 1-trial_sides[elapsed_trials]
                correct = (trial_sides[elapsed_trials] = 0)
                end_experiment(correct)
            end if
        end if

        ' Don't hog the CPU
        DelayMS(10)
    wend
    StopUnitRecording

    'Print summary stats
    print #1, ";"
    print #1, "Completed;", elapsed_trials, ";Correct;", num_correct

    close #1 'Close the file
end sub