'''Created by Sean Martin on 06/08/2019 for Gao Xiang Ham'''
'''Contact martins7@tcd.ie'''
'''Please note that lever input signals are inverted'''

'''Declare global variables'''

' Number of trials
dim num_trials
dim elapsed_trials

' Store per trial information
dim trial_sides
dim trial_rewards

'timing delays
dim fi_delay
dim trial_delay

'experiment value
dim fr_value
dim fr_count

'input pins
dim left_lever_inpin
dim right_lever_inpin
dim nosepoke_inpin

'output pins
dim left_lever_outpin
dim right_lever_outpin
dim left_light_outpin
dim right_light_outpin
dim house_light_outpin
dim food_outpin

'store input values
dim left_lever_value
dim right_lever_value

'store state
dim left_lever_active
dim right_lever_active
dim experiment_state
dim left_experiment

'timing
dim start_time
dim elapsed_time

' Optional experimenter tag
dim tag


'''Varable init routines'''

sub setup_pins()
    'setup variables to pins
    left_lever_inpin = 1
    right_lever_inpin = 2

    left_lever_outpin = 1
    right_lever_outpin = 2
    left_light_outpin = 9
    right_light_outpin = 10
    house_light_outpin = 12 'TODO get exact value
    sound_outpin = 13 'TODO get exact value
    fan_outpin = 14 'TODO get exact value
    food_outpin = 16
end sub

sub init_vars()
    'init variables such as trials and times
    elapsed_trials = 0
    start_time = 0
    elapsed_time = 0
    left_lever_active = 0
    right_lever_active = 0
end sub

sub init_arrays()
    ' Setup arrays which hold info for each trial
    dim indices
    indices = [0, num_trials-1]
    trial_sides = vararraycreate(indices, 12)
    trial_rewards = vararraycreate(indices, 12)

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

sub deliver_reward()
    ' Drop a food pellet
    trial_rewards[elapsed_trials] = trial_rewards[elapsed_trials] + 1
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

function check_shuffle(in_array, array_len)
    dim i
    dim temp
    check_shuffle = False
    for i = 0 to array_len - 3
        temp = in_array[i] + in_array[i+1] + in_array[i+2]
        if (temp = 0) or (temp = 3) then
            check_shuffle = True
        end if
    next
end function


'''Experiment Control Routines'''

sub new_experiment(first)
    ' Begin a new trial
    dim side
    dim side_nice
    if (first = 0) then
        elapsed_trials = elapsed_trials + 1
    end if

    SignalOut(sound_outpin) = 1
    DelayMS(5000)
    SignalOut(sound_outpin) = 0
    fr_count = 0

    if (elapsed_trials < num_trials) then
        if (trial_sides[elapsed_trials] = 1) then
            set_left_side(on)
            left_experiment = True
            side = ";FI;"
            side_nice = "fixed interval"
        else
            set_right_side(on)
            left_experiment = False
            side = ";FR;"
            side_nice = "fixed ratio"
        end if
        print "Starting Trial number ", elapsed_trials+1, " Out of ", num_trials
        print "Showing the subject ", side_nice
        print #1, "Trial;", elapsed_trials+1
        print #1, "Type;", side
        print #1, "Begin;", TrialTime

        start_time = TrialTime
        experiment_state = "Start"
    end if
end sub

sub end_experiment()
    reset_left_right()
    print "Ending trial, num_rewards in this trial: ", trial_rewards[elapsed_trials]
    print #1, "End;", TrialTime
    print #1, ";"
    new_experiment(0)
end sub

sub full_init_before_record()
    ' Perform script init before recording
    dim i

    ' Print the csv file header
    print #1, tag, ";", num_trials, ";", fi_delay, ";", fr_value
    print #1, ";"

    ' Convert some delays to ms
    trial_delay = trial_delay * 1000 * 60
    Randomize 'Seed the rng based on the system clock

    'Setup the pins and turn on essential outputs
    setup_pins()
    reset()
    init_vars()
    init_arrays()
    SingalOut(fan_outpin) = on
    SignalOut(house_light_outpin) = on

    ' Shuffle trial sides
    dim bad
    bad = True
    while(bad)
        knuth_shuffle(trial_sides, num_trials)
        bad = check_shuffle(trial_sides, num_trials)
    wend

    print "Trial order is (0 FR 1 FI):"
    for i = half_point to num_trials - 1
        print trial_sides[i]
    next
end sub


''' Main subroutine follows this'''

sub main()
    ' Run the experiments and record the data
    '''NB Change important variables here'''
    num_trials = 6 'Number of trials is usually fixed 6
    trial_delay = 5 'How long between trials in minutes
    fi_delay = 30 'How long fi delay is in seconds
    fr_value = 6 'Number of FR presses needed
    tag = "Optional tag" ' You may tag this experiment
    'output in the same name format as other axona files
    open "data.log" for output as #1

    full_init_before_record()
    StartUnitRecording
    new_experiment(1)

    ' Loop the recording for the number of trials
    while (elapsed_trials < num_trials)

        ' End trial
        if (TrialTime - start_time > trial_delay) then
            end_experiment()
        elseif (left_lever_active = 1) and (SignalIn(left_lever_inpin) = off) and (experiment_state = "Start") and (trial_sides[elapsed_trials] = 1) then
            deliver_reward()
            experiment_state = "Reward"
        elseif (right_lever_active = 1) and (SignalIn(right_lever_inpin) = off) and (experiment_state = "Start") and (trial_sides[elapsed_trials] = 0) then
            fr_count = fr_count + 1
            if (fr_count = fr_value)
                deliver_reward()
                experiment_state = "Reward"
                fr_count = 0
            end if
        elseif (experiment_state = "Reward") and (SignalIn(nosepoke_inpin) = on) then
            if (trial_sides[elapsed_trials] = 1) then
                DelayMS(fi_delay*1000)
            end if
            experiment_state = "Start"
        end if

        ' Don't hog the CPU
        DelayMS(10)
    wend
    StopUnitRecording

    close #1 'Close the file
end sub