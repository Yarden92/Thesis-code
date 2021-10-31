### the input in time

<img src="https://latex.codecogs.com/svg.latex?\Large q(t)=\frac{2.2}{cosh(t)}"/> 

### the input in &xi; domain
<img src="https://latex.codecogs.com/svg.latex?\Large c(\xi)=\frac{0.587783}{cosh(\pi\xi )} \cdot e^{-\pi j}"/> 

### params:
* <img src="https://latex.codecogs.com/svg.latex? t \in [-15:15] : (N_t=1024)"/> 
* <img src="https://latex.codecogs.com/svg.latex? \xi \in [-53:53] : (N_\xi=2048)"/> 


### the INFT transform

<img src="https://latex.codecogs.com/svg.latex?\Large c(\xi) \overset{INFT}{\longrightarrow} \hat{q}(t) : q(t)=\hat{q}(t)"/> 

#### params of the INFT
* "discretization": 4
* "contspec_type": 1
* "contspec_inversion_method": 0
* "discspec_type": 0
* "max_iter": 100
* "oversampling_factor": 8
