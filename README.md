# CHIanalysis
This is a set of tools for analyzing behavioral data from head injured animals.
Specifically, head injured and sham animals performed a beam-walk at different 
time-points post-injury.  Limb position during experimental trials was extracted 
using [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut). These scripts
analyze that limb position data, identifying "foot falls" and computing
associated features like foot fall frequency and total foot fall abisition (the 
integral of y-position during foot falls over time). 

To generate visualizations of foot fall absition during beam walk 
in control and head injured animals at two time points post-injury,
run 'python3 -i absition_examples.py'.