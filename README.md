# Simulation-of-electrons-in-crystals
A script to simulate lightwave driven dynamics of electrons in solid state crystals according to the semiclassical Bloch Equations

Features 3 classes:
  - Transient: Used to add a lightwave transient to the simulation
  - Bandstructure: Used to add a Bandstructure to the simulation
  - Solution: This is th "final" object. Initialise this with a Transient object and a Bandstructure object as arguments
    - call solve()
    - call process()
    --> All the results can now be extracted from the solution object :)
