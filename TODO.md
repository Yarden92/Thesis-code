# TODO list:
* validate that the pulse shape is fine
* de-pulse shaping / sampling: how to convert from pulse back to (i,q) modulation symbols?
* the BW is made up empirically, is there a formula way to calculate it?

# Tasks that are done:
* based on the T -> calc the max XI boundaries
* find the xi boundaries by plotting and check if data wasn't trimmed
* find the time boundaries by plotting and check if data wasn't trimmed
* skip the bounce state to make it run faster
* TODO: 3) compare with IFFT - on small factors they should be similar (on abs) up to time scale (pi) 
* missing upsampling
* do linear IFFT and check the width of the signal in time (< but there is no units on t axis..)
* pad x3i with zeros before and after
* send stas and bella the scholarship documents
* make the vectors more generic so we won't need to make sure that: `len(X_xi)+len(pulse_shaper)=2048`

# what I've done lately:
* I've found an interesting example for the pulse shaping, I adopted some of it to get better shape of the pulse
* added an upsampling (zero padding between each sample)
* important discovery: D length (time domain length) must be a power of 2! 
  * I updated the N_time fetching method accordingly
* empirically discovery: BW should be about 700&bullet;2&pi; for the signal after NFT to look like before INFT