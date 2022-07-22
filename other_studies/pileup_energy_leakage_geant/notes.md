# Energy Leakage from Pileup

A 3 GeV electron will have a higher amount of leakage out the back of a calorimeter than 2 1.5 GeV electrons. Some questions:

* Does this effect our pileup?
* How will this bias our pileup correction?
* Can we apply a correction for this bias?

Looking at David S. thesis for some initial answers... Not seeing anything initially.

Lets start by setting up a simple toy geant model and shooting in some electrons. We already have the PIONEER Geant4 container, so this will be set up to work with Geant4 11.01.

Created a branch to work with: `https://github.com/PIONEER-Experiment/MonteCarlo/pull/new/feature/gm2_calo`

---

Ok, so now we have the data from the simulation.... but now what? Well, in the background, we need to refine the simulation to make sure the inputs are correct! If we aren't simulating the physics properly, we will get the wrong answer. However, in parallel we need to assess the impact of this effect on the energy reconstructed and determine how to correct it.

Also need to think about the effect of crystal truncation!

---

Can we create a ML model / interpolator to repair the energy loss... I think its likely, similar to the one used in PIONEER.
