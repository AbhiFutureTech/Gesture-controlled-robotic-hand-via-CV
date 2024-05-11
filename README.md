
Description
We made 2 key modifications to the original version:


Hand gesture recognition


We added the hand_tracking_cpu_main to make the system recognize hand gestures in real-time. To make this work, we employed hand gesture recognition calculators and made changes to the original .pbtxt graphs (see the latest commits).

Currently there are 2 versions of hand gesture calculcator:

HandGestureCalculator: rule-based hand gesture recognition. Inspired by the code from the TheJLifeX repo.

HandGestureCalculatorNN: neural network-based gesture recognition.

By default, HandGestureCalculator is used. Feel free to modify the hand_landmark_cpu.pbtxt graph to change the gesture calculator.

We used Jesture AI SDK (python/annotation.py) to collect the data for neural network training.

ZeroMQ message passing
ZeroMQ is a tool for message passing between different processes. It allows to communicate between e.g. a binary file compiled from C++ and a python script. In our code, we use the hand_tracking_cpu_main as a Requester and the zmq_server_demo.py as a Replier (see REQ-REP strategy).

To make all these things work we used the cppzmq header files (see examples/desktop/hand_tracking dir).
