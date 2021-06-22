# NemesisPy

In the Data/ folder, I have turned the data in gasinforef.dat into a Python dictionary object for easy look up and calculation of mean molecular weight. 
I have also included a mix of reference data to be used throughout the code. I could do the same for planet data and stellar data here. 

In the Profile/ folder, I have written some objects to hold input data.

In the Layer/ folder, I have pythonised the Radtran/Path routines to split an atmosphere into layers and calculate average layer properties. 
I have tested it against the Fortran code, and they agree well. 

There is a test_layer.py file in the root folder that shows how everything is tied together. 
