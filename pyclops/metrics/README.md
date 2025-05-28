Includes metrics by which you can evaluate your PyCLOPS & protein structures.

We likely want to add an OpenMM based way of scoring too?

And lastly, a way to take a structure which is run through the conditioning and then add the cyclic "ligand" it has collapsed into in the topology (based on the toy example?) -- This may be fairly complex.

We also probably want a module or something that helps us easily plot ramachandran, pdbs, etc.

We also probably want some ML based way of doing all of this, but I am dubious of this, as it will very likely never have seen any of this...

Alex had a great idea today -- to test how plausible a given cyclic sample is, just add the ligand into the pdb and relax it over say 5 steps...
