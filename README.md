# Deep Reinforcement Learning-CartPole
Double Q-Learning to play CartPole

This project is similar to <a href="https://github.com/ykteh93/Deep_Reinforcement_Learning-Atari">my other project</a> on Atari games, except double Q-learning is implemented here.
* Q-learning uses same values to select and to evaluate, while double Q-learning decouples that with two separate networks and update one at random for each experience. <a href="https://arxiv.org/pdf/1509.06461.pdf">The details can be read here.</a>

<br>
<dl>
  <dt>The details is as follow:</dt>
  <ul>
  <li>The task is to keep a pole balanced indefinitely.</li>
  <li>This environment has a 4 dimensional observations and two discrete actions (left, right).</li>
  <li>To investigate the effect of sparse rewards, reward is modified to be 0 on non-terminating steps and -1 on termination.</li>
  <li>input &rarr; hidden layer (100 units) + RELU &rarr; linear layer &rarr; state-action value function</li>
  </ul>
</dl>
