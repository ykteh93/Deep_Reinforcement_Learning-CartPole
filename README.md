# Deep Reinforcement Learning-CartPole
Double Q-Learning to play CartPole

This project is similar to <a href="https://github.com/ykteh93/Deep_Reinforcement_Learning-Atari">my other project</a> on Atari games, except double Q-learning is implemented here.
* Q-learning uses same values to select and to evaluate, while double Q-learning decouples that with two separate networks and update one at random for each experience. <a href="https://papers.nips.cc/paper/3964-double-q-learning.pdf">The details can be read here.</a>
<br>
<p align="center"> 
<img src="https://github.com/ykteh93/Deep_Reinforcement_Learning-CartPole/blob/master/Graphs_and_Figure/For_README.png">
</p>

<br>
<dl>
  <dt>The details of this project is as follow:</dt>
  <ul>
  <li>The task is to keep a pole balanced indefinitely.</li>
  <li>This environment has a 4 dimensional observations and two discrete actions (left, right).</li>
  <li>To study the effect of sparse rewards, default reward is modified to be 0 on non-terminating steps and -1 on termination.</li>
  <li>Architecture: input &rarr; hidden layer (100 units) + RELU &rarr; linear layer &rarr; state-action value function</li>
  <li>The saved model after training are included (Run the Load_Model.py file to evaluate).</li>
  </ul>
</dl>
