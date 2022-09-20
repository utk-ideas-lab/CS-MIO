# CS-MIO
Compressive-sensing assisted mixed-integer optimization (CS-MIO) is a solver for data driven discovery of dynamical system from highly noisy data.

## Source Code
This contains the Python source code for discovering the exact governing equations from highly noisy data.

- **DataGenerator**: This folder contains the data generator for the studied dynamcal system examples.
- **Optimizer**: The CS_MIO optimizer is included in this folder for training to obtain the governing equations of dynamical systems.
- **test_xx**: These are the testing code to obtain the goverining equations for a specific dynamical system by directly running it.
- **docs**: The resulting figures for the learned models.


The time series data of state variables $\mathbf{x}$ are collected in a time instants $t_1,t_2,\dots,t_N$ and organize into two matrices $\mathbf{X}$ and $\dot{\mathbf{X}}$. The augmented library matrix $\boldsymbol{\Theta}(\mathbf{X})$ is established according to the constructed term library. We then apply the CS-MIO algorithm to uncover each equation.
The main entry of the CS-MIO optimizer is to set the parameters of the method as below:

    def __init__(
                self,
                dimension=3,              # dimension of the dynamical system
                order=2,                  # polynomial order of the constructed terms in $\boldsymbol{\Theta}(\mathbf{X})$
                num_candidate_terms=100,  # number of the candidate terms reamined after the compressive sensing algorith
                lasso_alpha=0.000001,     # regularization value for the compressive sensing
                intercept=True,           # whether or not to model the intercept in the model
                term_ks=[],               # number of nonzero terms specified in the equations as a vector
                betas_ub=1000,            # the upper bound for solving the MIO problem 
                betas_lb=-1000,           # the lower bound for solving the MIO problem
                time_limit=600,           # time limit for solving the MIO problem
                mip_gap=0.0,              # the gap of the solution to the best lower bound. 
                                          # Either time_limit or mip_gap satisfication will trigger stopping of the MIO algorithm.
                mip_detail=False,         # whether to show the details of the MIO solving procedure.
        ):

## Example
### Lorenz3 system
We consider to collect data from the below 3-dimensional chaotic Lorenz system governed by the following equations:

$\dot{x} = \alpha (y-x),$

$\dot{y} = x (\rho - z) - y,$

$\dot{z} = x y - \beta z.$

In this example, to obtain the governing equations of Lorenz 3 dynamical system, we run the below code by first generating the data, then applying the CS-MIO algorithm, and finally plotting the leanred model in figure.
    
    # Generate training data
    x, y, snr = lorenz3.lorenz3generator(seed=40, noise_type=1, data_noise=300, end_time=60)
    print('SNR: %.4f' % snr)

    # Discover equations. We define the system dimension to be 3. We use the fifth-order polynomial terms to construct the library. 
    # We can manualy set the number of nonzero terms in the governing equations using a vector [2,3,2] for Lorenz 3 system. Or we can obtain this by the tuning             # procedure by cross validation in the paper. 
    model = CSMIO(optimizer=CSMIOOptimizer(dimension=3, order=5, term_ks=[2, 3, 2]))
    model.fit(x, y)

    # show the trajectory of identified equations
    lorenz3.plot_trajectory(model)

<img src="./docs/Lorenz3_Gaussian_300_CSMIO.PNG" width="300">

### Lorenz 96
