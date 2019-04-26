sub can_print()
    print #1, "Wow I can print in here!"
end sub

sub main()
    dim last_time
    dim time_since
    dim experimenttime

    experimenttime = 10 'How long to run the experiment for
    open "data.log" for output as #1
    StartUnitRecording

    print #1, "Testing file print!"
    can_print()
    print #1, "Did that print?"
    last_time = TrialTime

    while (TrialTime/1000 < experimenttime)
        time_since = TrialTime - last_time
        if (time_since > 1000) then
            print #1, "Testing file print!"
            can_print()
            print #1, "Did that print?"
            last_time = TrialTime
        end if
        DelayMS(10)
    wend
    StopUnitRecording
end sub