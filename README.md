# Deep Reinforcement Learning-CartPole
Double Q-Learning to play CartPole

This project is similar to <a href="https://github.com/ykteh93/Deep_Reinforcement_Learning-Atari">my other project</a> on Atari games, except double Q-learning is implemented here.

Q-learning uses same values to select and to evaluate, while double Q-learning decouples that with two separate network. <a href="https://arxiv.org/pdf/1509.06461.pdf">The details motivation can be read here.</a>

The key details of this project is as follow.
<dl>
  <dt>State Space:</dt>
  <ul>
  <li>Environment observation is converted to greyscale and reduced in size (60 x 60) to conserve memory.</li>
  <li>4 consecutive frames are stacked together (60 x 60 x 4) in order to capture the motion.</li>
  </ul>
</dl>
