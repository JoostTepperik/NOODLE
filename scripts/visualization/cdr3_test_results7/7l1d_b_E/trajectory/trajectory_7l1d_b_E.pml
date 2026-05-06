# Trajectory: 7l1d_b_E  (51 frames)
# Loop resi 3-21

load trajectory_7l1d_b_E.pdb, traj
bg_color black
set stick_radius, 0.15
set stick_ball, on
set stick_ball_ratio, 1.5
hide everything, traj
show sticks, traj
color grey60, traj
spectrum b, blue_white_red, traj and resi 3-21, minimum=0, maximum=30

# Per-frame labels via pseudoatoms (top-left corner)
set label_relative_mode, 1
pseudoatom lbl, pos=[0,0,0], state=1, label="step 0 | E=109.17 | cl=38.2165A"
pseudoatom lbl, pos=[0,0,0], state=2, label="step 1 | E=101.13 | cl=24.9223A"
pseudoatom lbl, pos=[0,0,0], state=3, label="step 2 | E=109.85 | cl=11.3967A"
pseudoatom lbl, pos=[0,0,0], state=4, label="step 3 | E=125.47 | cl=30.8776A"
pseudoatom lbl, pos=[0,0,0], state=5, label="step 4 | E=136.34 | cl=9.7513A"
pseudoatom lbl, pos=[0,0,0], state=6, label="step 5 | E=143.24 | cl=22.6403A"
pseudoatom lbl, pos=[0,0,0], state=7, label="step 6 | E=145.32 | cl=14.9279A"
pseudoatom lbl, pos=[0,0,0], state=8, label="step 7 | E=145.91 | cl=4.7066A"
pseudoatom lbl, pos=[0,0,0], state=9, label="step 8 | E=144.65 | cl=7.2523A"
pseudoatom lbl, pos=[0,0,0], state=10, label="step 9 | E=144.91 | cl=7.7336A"
pseudoatom lbl, pos=[0,0,0], state=11, label="step 10 | E=147.70 | cl=10.0751A"
pseudoatom lbl, pos=[0,0,0], state=12, label="step 11 | E=146.86 | cl=1.8687A"
pseudoatom lbl, pos=[0,0,0], state=13, label="step 12 | E=146.16 | cl=9.7418A"
pseudoatom lbl, pos=[0,0,0], state=14, label="step 13 | E=147.43 | cl=2.1939A"
pseudoatom lbl, pos=[0,0,0], state=15, label="step 14 | E=148.42 | cl=6.8882A"
pseudoatom lbl, pos=[0,0,0], state=16, label="step 15 | E=148.49 | cl=2.2732A"
pseudoatom lbl, pos=[0,0,0], state=17, label="step 16 | E=148.90 | cl=5.5587A"
pseudoatom lbl, pos=[0,0,0], state=18, label="step 17 | E=149.61 | cl=2.6401A"
pseudoatom lbl, pos=[0,0,0], state=19, label="step 18 | E=149.19 | cl=5.3191A"
pseudoatom lbl, pos=[0,0,0], state=20, label="step 19 | E=148.86 | cl=2.3876A"
pseudoatom lbl, pos=[0,0,0], state=21, label="step 20 | E=148.36 | cl=3.4086A"
pseudoatom lbl, pos=[0,0,0], state=22, label="step 21 | E=146.52 | cl=3.8135A"
pseudoatom lbl, pos=[0,0,0], state=23, label="step 22 | E=145.74 | cl=3.5812A"
pseudoatom lbl, pos=[0,0,0], state=24, label="step 23 | E=143.80 | cl=3.8925A"
pseudoatom lbl, pos=[0,0,0], state=25, label="step 24 | E=143.76 | cl=1.1907A"
pseudoatom lbl, pos=[0,0,0], state=26, label="step 25 | E=143.08 | cl=2.4834A"
pseudoatom lbl, pos=[0,0,0], state=27, label="step 26 | E=141.62 | cl=1.9559A"
pseudoatom lbl, pos=[0,0,0], state=28, label="step 27 | E=140.32 | cl=2.5376A"
pseudoatom lbl, pos=[0,0,0], state=29, label="step 28 | E=139.11 | cl=2.2836A"
pseudoatom lbl, pos=[0,0,0], state=30, label="step 29 | E=138.62 | cl=1.7357A"
pseudoatom lbl, pos=[0,0,0], state=31, label="step 30 | E=137.60 | cl=2.5011A"
pseudoatom lbl, pos=[0,0,0], state=32, label="step 31 | E=136.24 | cl=0.7230A"
pseudoatom lbl, pos=[0,0,0], state=33, label="step 32 | E=135.18 | cl=2.1675A"
pseudoatom lbl, pos=[0,0,0], state=34, label="step 33 | E=134.55 | cl=1.4469A"
pseudoatom lbl, pos=[0,0,0], state=35, label="step 34 | E=133.05 | cl=2.0963A"
pseudoatom lbl, pos=[0,0,0], state=36, label="step 35 | E=132.51 | cl=5.4323A"
pseudoatom lbl, pos=[0,0,0], state=37, label="step 36 | E=130.54 | cl=7.5197A"
pseudoatom lbl, pos=[0,0,0], state=38, label="step 37 | E=129.78 | cl=3.7259A"
pseudoatom lbl, pos=[0,0,0], state=39, label="step 38 | E=130.12 | cl=4.5103A"
pseudoatom lbl, pos=[0,0,0], state=40, label="step 39 | E=130.81 | cl=7.6290A"
pseudoatom lbl, pos=[0,0,0], state=41, label="step 40 | E=130.14 | cl=1.7273A"
pseudoatom lbl, pos=[0,0,0], state=42, label="step 41 | E=130.90 | cl=3.7064A"
pseudoatom lbl, pos=[0,0,0], state=43, label="step 42 | E=133.22 | cl=4.9610A"
pseudoatom lbl, pos=[0,0,0], state=44, label="step 43 | E=134.59 | cl=4.6752A"
pseudoatom lbl, pos=[0,0,0], state=45, label="step 44 | E=134.50 | cl=4.3284A"
pseudoatom lbl, pos=[0,0,0], state=46, label="step 45 | E=134.08 | cl=3.9402A"
pseudoatom lbl, pos=[0,0,0], state=47, label="step 46 | E=133.50 | cl=6.3822A"
pseudoatom lbl, pos=[0,0,0], state=48, label="step 47 | E=134.47 | cl=8.1016A"
pseudoatom lbl, pos=[0,0,0], state=49, label="step 48 | E=134.07 | cl=5.6513A"
pseudoatom lbl, pos=[0,0,0], state=50, label="step 49 | E=134.73 | cl=2.8824A"
pseudoatom lbl, pos=[0,0,0], state=51, label="step 249 | E=130.82 | cl=0.0833A"

hide everything, lbl
show labels, lbl
set label_color, white, lbl
set label_size, -0.8
set label_position, [-25, 20, 0]
set label_font_id, 7

mset 1 -51
set movie_fps, 4
zoom traj
mplay
