# Explaining the study in five minutes (for a student presentation)

**The question.** If a building changes (more people, hotter weather), does a smart controller that keeps correcting its own temperature-prediction model do better than one that never updates it?

**The setup.** We did not use a real building. We built a small computer model of one classroom and treated it as the "real" room. A second, simpler model, the *digital twin*, tries to predict the room temperature 15 minutes ahead using five inputs: current room temperature, outside temperature, number of people, sunlight and cooling power. We deliberately made the room more complicated than the twin (cooling efficiency drops in hot weather; more people means more fresh air), so the twin is never perfect.

**Four controllers.** (1) A thermostat. (2) A twin that learns once from three days of data and then never changes. (3) A twin that corrects itself only when its recent prediction errors become noticeably large (the *gated* twin). (4) A twin that corrects itself every step.

**Four days-of-trouble.** Normal day; twice as many people from 12:00 to 16:00; outdoor temperature up by 5 degrees; both together. Each was repeated with 20 different random seeds (different weather noise, sensor noise, etc.).

**What we found.** When it got hotter outside, correcting the twin reduced prediction errors and kept the room closer to its target range, using about 0.3% more energy. With more people only, or on a normal day, correcting did not help; correcting every step was slightly worse. The improvements are real across seeds but tiny (hundredths of a degree), smaller than the 0.1 degree sensor noise.

**Why you should be careful.** The room is invented, its numbers are assumptions, and the thermostat is a crude comparison. So this shows how the method behaves in a controlled test, not how it would behave in a real building.

**Likely questions and short answers.**
* *Why not reinforcement learning?* Existing reviews say it needs lots of data and is rarely used in real buildings; we wanted something we can inspect and explain.
* *Is this new?* Updating building models online is not new. Our contribution is a careful, open comparison of *when* to update.
* *Why is the thermostat worse on energy?* Mostly its simple on/off-style logic and coarse 15-minute sampling, not because the AI is "smart".
* *Could the result be luck?* We repeated everything for 20 seeds and tested paired differences; the outdoor-temperature gains were consistent, but the tests are exploratory.
* *Does the twin see the future?* No. We tested this directly: changing future inputs does not change earlier decisions.
