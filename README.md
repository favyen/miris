MIRIS: Fast Object Track Queries in Video
=========================================

This is the implementation of the MIRIS query-driven tracking approach proposed
in ["MIRIS: Fast Object Track Queries in Video" (SIGMOD 2020)](https://favyen.com/miris/).

MIRIS is a query processing engine for large video datasets. It optimizes
queries with object track predicates, i.e., predicates evaluated over the
trajectory of an object as it moves through the camera frame during the segment
of video in which it is visible. These queries include selecting objects that
move from one region of the camera frame to another (e.g., find cars that turn
right through a junction) and selecting objects with certain speeds (e.g., find
animals that stop to drink water from a lake).


Requirements
------------

Python requirements are listed in `models/gnn/requirements.txt`. The Python
code should work on both 2.7+/3.6+. You will see the Go requirements after
trying to run some of the code. `ffmpeg` also needs to be installed for data
preparation.


Preparing Data
--------------

The YTStream dataset from the paper includes four of the sources of video data
used in the paper: UAV, Tokyo, Warsaw, and Resort. More details on the dataset are
available at https://favyen.com/miris/.

To get started:

	git clone https://github.com/favyen/miris.git
	cd miris
	wget https://favyen.com/miris/ytstream-dataset.zip
	unzip ytstream-dataset.zip
	python data/extract.py

This will create three subdirectories in data/, one for each dataset. Inside
each dataset directory, there are three directories, for example:

* data/beach/frames: contains the video frames, extracted as JPG
* data/beach/json: contains the object tracks computed via a baseline object
  tracking algorithm for the training and validation video segments. Also
  contains the object detections in the test segment. In a real system the
  object detector should be run on-the-fly.
* data/beach/videos: the original videos from the zip file, you can delete it
  after running `data/extract.py`.

The JPG files must be numbered like 000001.jpg, 000002.jpg, etc. The object
tracks and detections are in JSON files, and you can check `miris/detection.go`
to see the format.


Training Models
---------------

Next we need to train all of the models that MIRIS will use, except the object
detector, which this implementation assumes has already been executed.
Specifically, we need to train the GNN tracking model, which tries to
approximate the baseline tracking algorithm while reading fewer object
detections, and the RNN filtering and refinement models.

For example, to train the models for the shibuya query (called Tokyo dataset in
the paper) at 10/16 fps (i.e., looking at one in every 16 frames), we would:

	mkdir -p logs/shibuya/gnn
	mkdir -p logs/shibuya/16/{filter,refine}-rnn
	go run prepare_rnn.go shibuya 16
	cd models/gnn
	python train.py shibuya
	cd ../rnn
	python train.py ../../logs/shibuya/16/filter_rnn_ds.json ../../logs/shibuya/16/filter-rnn/model
	python train.py ../../logs/shibuya/16/refine_rnn_ds.json ../../logs/shibuya/16/refine-rnn/model

The following query choices are supported: uav (Q1), shibuya (Q2), warsaw (Q3),
shibuya-crosswalk (Q5), beach-runner (Q6), and warsaw-brake (Q7). Q4 involves a
different video data source (Berkeley DeepDrive) that needs to be obtained
separately.


Planning
--------

Now we can execute the planner. The planner will decide what parameters to use
for filtering, uncertainty resolution, and tracking. To decide the base
sampling framerate, you need to run the planner multiple times at different
framerates.

Suppose we want to run the planner at alpha=0.9 (>=90% precision and recall)
and 10/16 fps. Then:

	mkdir -p logs/shibuya/16/0.9
	go run plan.go shibuya 16 0.9

This produces a plan stored at `logs/shibuya/16/0.9/plan.json`.


Execution
---------

Once we have the plan file, we can execute MIRIS over a large dataset:

	go run exec.go shibuya logs/shibuya/16/0.9/plan.json

This produces several intermediate outputs in `logs/shibuya/16/0.9/`, which the
implementation will reuse on future runs in case an error happens. (If you
change something and want to re-execute, you may need to delete these
intermediates.) The final output object tracks are produced at `logs/shibuya/16/0.9/final.json`.


Evaluation
----------

You can compute precision, recall, and F1 score against the ground truth stored
in CSV files in data/ folder. For example:

	go run eval.go data/shibuya.csv logs/shibuya/16/0.9/final.json
