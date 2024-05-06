from cheftgp.gaussianprocess import TrainTestSplit

# creates the training and testing masks for observables plotted against angle

# Fullspaceanglessplit... objects spread training and testing points evenly across the whole input space
Fullspaceanglessplit = TrainTestSplit(
    "allangles", [6], [3], xmin_train_factor=[0], xmax_train_factor=[1]
)
Fullspaceanglessplit1 = TrainTestSplit(
    "allangles1", [5], [3], xmin_train_factor=[0], xmax_train_factor=[1]
)
Fullspaceanglessplit2 = TrainTestSplit(
    "allangles2", [4], [4], xmin_train_factor=[0], xmax_train_factor=[1]
)
# Forwardanglessplit... objects spread training and testing points evenly across the whole input space except omitted
# at backwardmost angles
Forwardanglessplit = TrainTestSplit(
    "forwardangles", [6], [3], xmin_train_factor=[0], xmax_train_factor=[5 / 6]
)
Forwardanglessplit1 = TrainTestSplit(
    "forwardangles1",
    [5],
    [3],
    xmin_train_factor=[0],
    xmax_train_factor=[1],
    xmin_test_factor=[0],
    xmax_test_factor=[4 / 5],
)
Forwardanglessplit2 = TrainTestSplit(
    "forwardangles2",
    [6],
    [3],
    xmin_train_factor=[0],
    xmax_train_factor=[5 / 6],
    xmin_test_factor=[0],
    xmax_test_factor=[5 / 6],
)
# Backwardanglessplit... objects spread training and testing points evenly across the whole input space except omitted
# at forwardmost angles
Backwardanglessplit = TrainTestSplit(
    "backwardangles", [6], [3], xmin_train_factor=[1 / 6], xmax_train_factor=[1]
)
Backwardanglessplit1 = TrainTestSplit(
    "backwardangles1",
    [5],
    [3],
    # xmin_train_factor=0,
    xmin_train_factor=[1 / 5],
    xmax_train_factor=[1],
    xmin_test_factor=[1 / 5],
    xmax_test_factor=[1],
)
Backwardanglessplit2 = TrainTestSplit(
    "backwardangles2",
    [6],
    [3],
    xmin_train_factor=[1 / 6],
    xmax_train_factor=[1],
    xmin_test_factor=[1 / 6],
    xmax_test_factor=[1],
)
# Middleanglessplit... objects spread training and testing points evenly across the whole input space except omitted
# at backwardmost and forwardmost angles
Middleanglessplit1 = TrainTestSplit(
    "middleangles1",
    [5],
    [3],
    # xmin_train_factor=0,
    # xmax_train_factor=1,
    xmin_train_factor=[1 / 5],
    xmax_train_factor=[4 / 5],
    xmin_test_factor=[1 / 5],
    xmax_test_factor=[4 / 5],
)
Middleanglessplit2 = TrainTestSplit(
    "middleangles2",
    [6],
    [3],
    xmin_train_factor=[1 / 6],
    xmax_train_factor=[5 / 6],
    xmin_test_factor=[1 / 6],
    xmax_test_factor=[5 / 6],
)

# traintestsplit_vsangle_array = [
#     Fullspaceanglessplit,
#     Forwardanglessplit,
#     Backwardanglessplit,
#     Forwardanglessplit2,
#     Backwardanglessplit2,
#     Fullspaceanglessplit1,
#     Fullspaceanglessplit2,
#     Middleanglessplit1,
#     Middleanglessplit2
# ]

# creates the training and testing masks for observables plotted against energy
# Nolowenergysplit... objects spread training and testing points evenly across the whole input space except omitted
# at low energy/momentum
Nolowenergysplit = TrainTestSplit(
    "nolowenergy",
    [3],
    [4],
    offset_train_min_factor=[100 / 350],
    xmin_train_factor=[100 / 350],
    offset_test_min_factor=[100 / 350],
    xmin_test_factor=[100 / 350],
    offset_train_max_factor=[-50 / 350],
    offset_test_max_factor=[-50 / 350],
)
Nolowenergysplit1 = TrainTestSplit(
    "nolowenergy1",
    [3],
    [4],
    offset_train_min_factor=[50 / 350],
    xmin_train_factor=[50 / 350],
    offset_test_min_factor=[50 / 350],
    xmin_test_factor=[50 / 350],
)
# Yeslowenergysplit... and Allenergysplit... objects spread training and testing points evenly across the whole input space
Yeslowenergysplit = TrainTestSplit(
    "yeslowenergy",
    [4],
    [4],
    offset_train_min_factor=[0],
    xmin_train_factor=[0.01],
    offset_test_min_factor=[0],
    xmin_test_factor=[0],
    offset_train_max_factor=[-50 / 350],
    offset_test_max_factor=[-50 / 350],
)
Allenergysplit = TrainTestSplit(
    "allenergy",
    [4],
    [4],
    offset_train_min_factor=[0],
    xmin_train_factor=[0],
    offset_test_min_factor=[0],
    xmin_test_factor=[0],
    offset_train_max_factor=[-50 / 350],
    offset_test_max_factor=[-50 / 350],
)
Allenergysplit1 = TrainTestSplit(
    "allenergy1", [5], [3], xmin_train_factor=[0], xmax_train_factor=[1]
)
Allenergysplit2 = TrainTestSplit(
    "allenergy2", [4], [4], xmin_train_factor=[0], xmax_train_factor=[1]
)
Midenergysplit = TrainTestSplit(
    "midenergysplit",
    [7],
    [3],
    xmin_train_factor=[1 / 7],
    xmax_train_factor=[6 / 7],
    xmin_test_factor=[1 / 7],
    xmax_test_factor=[6 / 7],
)
# traintestsplit_vsenergy_array = [
#     Nolowenergysplit,
#     Yeslowenergysplit,
#     Allenergysplit,
#     Allenergysplit1,
#     Allenergysplit2,
#     Midenergysplit,
# ]

All2Dsplit1 = TrainTestSplit(
    "all2d1", [4, 4], [3, 3], xmin_train_factor=[0, 0], xmax_train_factor=[1, 1]
)

# traintestsplit_2d_array = [
#     All2Dsplit1,
# ]
