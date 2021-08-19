# MiniCombust Documentation

## Documentation goals
In contrast to other MiniApps which abstract away the underlying physics and design choices and hide these details in papers which might be 
behind paywalls, we want it to be clear what MiniCombust does. We try to describe the physics in terms of the governing equations, 
choices of physical models (e.g. droplet atomization, turbulence, emissions modelling). We describe how we discretise and solve the
equations, and where source terms relate to the physics in the code. We also try to describe the algorithms we use (e.g. gradient calculation with slope limiting).

To be able to meaningfully present these, we've chosen to use Jupyter Notebooks, which allow us to mix equatins in Latex format with text, images,
and runnable code to clarify algorithms. Eventually we might want to combine these as comprehensive Sphinx documentation, or even [executable books](https://jupyterbook.org/intro.html),
and perhaps allow a demo version of MiniCombust to be launched in [Binder](https://mybinder.org/) or similar.

## Overview
* [Physics](./Physics.ipynb)
* [Algorithms](./Algorithms.ipynb)
* [Particle-flow interaction](./ParticleFlowInteraction.ipynb)
* [Communication](./Communication.ipynb)
* [Developer's Guide](./DevelopersGuide.ipynb)
* [Data Structures](./DataStructures.ipynb)
* [Notes about Dolfyn](./Dolfyn.ipynb), on which MiniCombust is heavily based
* [Mesh Loading Example](./MeshLoadingExample.ipynb)
* [Notes](./Notes.ipynb)
* [Resources](./Resource.ipynb)
