sub main()
     StartUnitRecording
     SignalOut(1) = on
     while (TrialTime/1000 < 100000)

          if (SignalIn(1) = on) then
               print "Pressed 1"
          elseif(SignalIn(2) = on) then
              print "Pressed 2"
          elseif(SignalIn(7) = on) then
              print "Pressed 7"
          elseif(SignalIn(8) = on) then
              print "Pressed 8"
          elseif(SignalIn(6) = on) then
              print "Pressed 6"
          elseif(SignalIn(4) = on) then
              print "Pressed 4"
          end if

          ' Don't hog the CPU
          DelayMS(10)
     wend
     StopUnitRecording
end sub
