' Experiment time set here
dim experimenttime

'input pins
dim lever1inpin
dim lever2inpin

'output pins
dim lever1outpin
dim lever2outpin
dim light1outpin
dim light2outpin
dim foodoutpin

'store input values
dim lever1value
dim lever2value

'store state
dim lever1active
dim lever2active
dim completed1
dim completed2

'setup variables to pins
sub setup_pins()
     lever1inpin = 1
     lever2inpin = 2

     lever1outpin = 1
     lever2outpin = 2
     light1outpin = 9
     light2outpin = 10
     foodoutpin = 16
end sub

' resets all digital outputs to 0
sub reset()
     SignalOut(all) = 0
     lever1active = 0
     lever2active = 0
end sub

sub set_first_side(state)
     SignalOut(light1outpin) = state
     SignalOut(lever1outpin) = state

     lever1active = state
end sub

sub set_second_side(state)
     SignalOut(lever2outpin) = state
     SignalOut(light2outpin) = state

     lever2active = state
end sub

' Do the initial setup
sub init_experiment()
     if (Rnd(1) < 0.5) then
          set_first_side(on)
     else
          set_second_side(on)
     end if

     completed1 = 0
     completed2 = 0
end sub

sub drop_food()
     SignalOut(foodoutpin) = on
     DelayMS(15)
     SignalOut(foodoutpin) = off
end sub

sub main()
     experimenttime = 60 'How long to run the experiment for

     Randomize 'Seed the rng based on the system clock

     setup_pins()
     reset()
     init_experiment()

     StartUnitRecording
     while (TrialTime/1000 < experimenttime)

          lever1value = SignalIn(lever1inpin)
          lever2value = SignalIn(lever2inpin)

          if (lever1value = off) and (lever1active = 1) then
               print "pressed first side"
               set_first_side(off)
               if (completed2 = 0) then
                    DelayMS(200)
                    set_second_side(on)
               end if
               completed1 = 1
          end if

          if (lever2value = off) and (lever2active = 1) then
               print "pressed second side"
               set_second_side(off)
               if (completed1 = 0) then
                    DelayMS(200)
                    set_first_side(on)
               end if
               completed2 = 1
          end if

          if (completed1 = 1) and (completed2 = 1) then
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