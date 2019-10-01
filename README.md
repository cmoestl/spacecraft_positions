SPACECRAFT_POSITIONS
====================

This python code makes the movies for the positions of the spacecraft Parker Solar Probe, STEREO-A, Bepi Colombo, and Solar Orbiter (assuming a February 2020 launch) as well as Mercury, Venus and Earth.

by C. Moestl/IWF-helio, Austrian Academy of Sciences, Graz, Austria.

https://www.iwf.oeaw.ac.at/en/user-site/christian-moestl/

last update: March 2019

Uses the python packages heliopy 0.7.0, sunpy 1.0.3, seaborn 0.9.0, numba 0.43.1, pickle


Two animations are available on youtube:

version with heliocentric intertial system (HCI)
https://www.youtube.com/watch?v=UZ0ISGJXA_M&t=73s

version with Earth-fixed (HEEQ)
https://www.youtube.com/watch?v=0ybvOYEl9VU&t=80s


Usage
-----
    git clone https://github.com/cmoestl/spacecraft_positions
    cd spacecraft_positions
    python sc_positions.py

The code produces a few plots and an animation of 100 frames starting with the launch of Parker Solar Probe in August 2018. For producing the animation, ffmpeg (https://ffmpeg.org/download.html) is assumed to be available on the command line.








