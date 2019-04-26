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
dim responses
dim delay_times
'TODO could be expanded to have other timings

'timing delays
dim max_match_delay
dim trial_delay
dim wrong_delay
dim reward_delay

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

'store a string to print
dim print_value

'timing
dim start_time
dim elapsed_time
dim back_time


'''Varable init routines'''

sub setup_pins()
    'setup variables to pins
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

sub init_vars()
    'init variables such as trials and times
    elapsed_trials = 0
    num_correct = 0
    print_value = "Empty"
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
    responses = vararraycreate(indices, 12)
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

sub deliver_reward()
    ' Drop a food pellet
    SignalOut(food_outpin) = on
    DelayMS(15)
    SignalOut(food_outpin) = off
end sub


'''Maths related subroutines'''

sub swap(byref a, byref b)
    dim temp
    temp = a
    a = b
    b = temp
end sub

function generate_random_float(max)
    ' Generate a random number in the range of 0 to max
    generate_random_float = trunc(Rnd(1) * (max + 1))
end function

sub knuth_shuffle(byref in_array, array_len)
    'Randomly shuffle an array - see
    'https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    dim i
    dim j
    for i = array_len - 1 to 1 step -1
        j = generate_random_float(i)
        swap(in_array[j], in_array[i])
    next
end sub

sub generate_delays()
    ' Generate a delay for each trial before the back lever extends
    ' In the range 1 to max_match_delay
    dim i
    for i = 0 to num_trials - 1
        delay_times[i] = 1 + generate_random_float(max_match_delay-1)
    next
end sub

'''Experiment Control Routines'''

sub init_experiment()
    ' TODO print value is not necessary if file 1 is accessible outside of main
    ' TODO might need fixed ratio between random samples
    ' Do the initial setup
    if (trial_sides[elapsed_trials] = 1) then
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

sub show_back_lever()
    ' Delay for a while and then show the back lever
    dim delay
    delay = generate_random_float(max_match_delay)
    DelayMS(delay * 1000)
    set_back_side(on)
    back_time = TrialTime
    experiment_state = "Match"
end sub

sub incorrect_response()
    ' Turn off the house light before starting next trial
    responses[elapsed_trials] = 0
    set_right_side(off)
    set_left_side(off)
    SignalOut(house_light_outpin) = off
    DelayMS(wrong_delay)
    SignalOut(house_light_outpin) = on
    DelayMS(wrong_delay)
end sub

sub correct_response()
    ' Give the primate some reward for being smart
    responses[elapsed_trials] = 1
    num_correct = num_correct + 1
    set_right_side(off)
    set_left_side(off)
    DelayMS(reward_delay)
    deliver_reward()
    DelayMS(trial_delay - reward_delay)
end sub

sub new_experiment()
    ' Begin a new trial
    elapsed_trials = elapsed_trials + 1
    if (elapsed_trials <= num_trials) then
        init_experiment()
    end if
end sub

sub full_init_before_record()
    'Perform script init before recording

    'convert delays to ms
    trial_delay = trial_delay * 1000
    wrong_delay = wrong_delay * 1000
    Randomize 'Seed the rng based on the system clock
    setup_pins()
    reset()
    init_vars()
    init_arrays()
    generate_delays()
    knuth_shuffle(trial_sides, num_trials)
    SignalOut(house_light_outpin) = on
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

    full_init_before_record()
    'this should output in the same name format as other axona files
    open "data.log" for output as #1
    StartUnitRecording
    init_experiment()
    print #1, print_value

    ' Loop the recording for the number of trials
    while (elapsed_trials <= num_trials)

        ' Wait for the primate to start by hitting a lever
        if (experiment_state = "Start") then

            if (left_lever_active = 1) then
                left_lever_value = SignalIn(left_lever_inpin)
                if (left_lever_value = off) then
                    elapsed_time = TrialTime - start_time
                    print #1, "Pressed left start lever after;", elapsed_time
                    set_left_side(off)
                    show_back_lever() ' Delay the back lever protrusion
                end if
            elseif (right_lever_active = 1) then
                right_lever_value = SignalIn(right_lever_inpin)
                if (right_lever_value = off) then
                    elapsed_time = TrialTime - start_time
                    print #1, "Pressed right start lever after;", elapsed_time
                    set_right_side(off)
                    show_back_lever() ' Delay the back lever protrusion
                end if
            end if
        elseif (experiment_state = "Match")
            if (back_lever_active = 1) then
                back_lever_value = SignalIn(back_lever_inpin)
                if (back_lever_value = off) then
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

                if (right_lever_value = off) then
                    elapsed_time = TrialTime - back_time
                    reset_left_right()
                    if (trial_sides[elapsed_trials] = 0) then
                        print #1, "Incorrect response after;", elapsed_time
                        incorrect_response()
                    else 'Correct response
                        print #1, "Correct response after;", elapsed_time
                        correct_response()
                    end if
                    new_experiment()
                    print #1, print_value
                elseif (left_lever_value = off) then
                    elapsed_time = TrialTime - back_time
                    reset_left_right()
                    if (trial_sides[elapsed_trials] = 1) then
                        print #1, "Incorrect response after;", elapsed_time
                        incorrect_response()
                    else 'Correct response
                        print #1, "Correct response after;", elapsed_time
                        correct_response()
                    end if
                    new_experiment()
                    print #1, print_value
                end if
            end if
        end if

        ' Don't hog the CPU
        DelayMS(10)
    wend
    StopUnitRecording

    'Print summary stats
    print #1, "Number of trials;", elapsed_trials, ";Number correct;", num_correct
    print #1, "Lever order (Left is 1);", trial_sides
    print #1, "Trial responses (Correct is 1);", responses
    print #1, "Non Match Delay (Seconds);", delay_times

    close #1 'Close the file
end sub