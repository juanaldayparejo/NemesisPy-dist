The NEMESIS code includes different retrieval techniques to solve the inverse problem: based on the observed electromagnetic spectrum of a given planetary atmosphere, we aim to find a set of atmospheric parameters that explain the observation.

Optimal Estimation
-----------------------------


Nested Sampling
-----------------------------


Model parameterisations
-----------------------------

The different retrieval parameterisations in the NEMESIS code are identified by a number of Variable IDs and Variable Parameters. These 

Adding custom parameterisations
---------------------------------

It is possible to add custom parameterisations to the retrieval scheme. In order to do so, one must implement the following steps:

1. Select one model parameterisation ID number, considering the numbers already taken by others.
2. Open the file for the class Variables_0.py, and change the function read_apr() to tell NEMESIS how to read this parameterisation from the .apr file. In this function, different fields must be properly filled:
        - Variables.VARIDENT : This field represents the Variable IDs, unique for this model parameterisation.
        - Variables.VARPARAM : This field represents any extra parameters the code needs to read to properly represent the parameterisation, but that will not be retrieved (i.e., they are not included in the state vector).
        - Variables.XA : This field represents the state vector and must be filled accordingly based on the parameters stored on the .apr file.
        - Variables.SA : This field represents the a priori covariance matrix and must be filled accordingly based on the parameters stored on the .apr file.
        - Variables.LX : This field tells whether a particular element of the state vector is carried in log-scale.
        - Variables.NUM : This field tells whether the jacobian matrix for this particular element of the state vector can be computed analytically (0) or numerically (1).
3. Open the file for the class Variables_0.py, and change the function calc_NXVAR() to tell NEMESIS the number of parameters in the state vector associated with this particular model parameterisation.
4. Open the file subprofretg.py, which maps the variables of the state vector to the different classes included in the forward model (e.g., Atmosphere, Surface...), as well as the gradients in the case that these are calculated analytically.

