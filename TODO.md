# TODO list:
* make the vectors more generic so we won't need to make sure that: `len(X_xi)+len(pulse_shaper)=2048`
* from 4.10:
  * TODO: 3: padd x3i with zeros before and after
  * send stas and bella the scholarship documents


# Tasks that are done:
* based on the T -> calc the max XI boundaries
* find the xi boundaries by plotting and check if data wasn't trimmed
* find the time boundaries by plotting and check if data wasn't trimmed
* skip the bounce state to make it run faster
* TODO: 3) compare with IFFT - on small factors they should be similar (on abs) up to time scale (pi) 
* missing upsampling
* do linear IFFT and check the width of the signal in time (< but there is no units on t axis..)


# Things I thought on:
* is the signal even band-limited? its totally random so maybe the bandwidth on freq domain is inf...
perhaps generating something else or bandpass filter it somehow
* check the XI params if they are wide enough to cover the entire signal

# what I've done lately:
* I've added an interesting example for the pulse shaping, try to mimick that and insert it to INFT
* if not as is, take the shape of the rrc from the example, and generate something similar by playing with the parameters 
  of the pulse shaping
* added an upsampling
* I'm still not sure how to find Tmax.. if I do IFFT on unknown vector, there is no time axis..