# PyGame Training GUI

This document describes the small graphical interface used for monitoring and
rating brain's motor cortex output. It replaces the early text-only terminal
concept with a PyGame window that integrates logging and feedback controls.

## Goals

1. Show motor cortex messages in a scrollable box on the left.
2. Include timestamps with date separators when the day changes.
3. Display the header `motor_cortex INFO` above the output list.
4. Provide colour coded rating buttons (-5 … 5) next to each message.
5. Allow selection of a message and submission of an alternate response for training.
6. Apply positive or negative reinforcement when a rating is chosen.
7. Training should propogate back through the layers of the network which are utilized to create a motor_cortex output and should produce a mapping where the output being trained is the entended output of the product of the auditory and visual context it had in its mind at that time.
8. The previous state may require an adjustable memory buffer for the internal states, which can be set in the config file (starting with 'training_buffer=30', which means 30 seconds, and should have a comment saying so.)

### Logging Handler

A custom `logging.Handler` will capture messages from the `motor_cortex` logger.  Each record is stored with its timestamp.  Dates are compared against the previous entry and a coloured divider is inserted when the day changes.

### Scrollable Output

Recent motor cortex messages are listed on the left. Older entries automatically
scroll off screen with the newest lines always visible.

### Rating Bar

A vertical column of clickable buttons from -5 to 5 appears on the right.  Each
click immediately calls ``MotorCortex.reinforce_output`` with the selected
rating for the latest token. Negative numbers weaken that output while positive
numbers reinforce it.
They also train the Inferior Frontal gyrus to suppress similar text for a
short period determined by ``ifg_feedback_buffer``.

### Correction Input

The right pane contains a small text box.  Entering text here and pressing Enter sends the string to a new `teach(text, context)` helper which encodes it via Wernicke's area and trains the motor cortex adapters using the current context embedding.  This effectively tells brain what it should have said.

### Persistence

Ratings and corrections are appended to `persistent/cli_feedback.log` so future sessions can replay them.  This keeps the manual guidance history intact.

## Integration

* `brain.py` now accepts `--gui_train`. When enabled it expands the PyGame
  viewer and attaches the `GUITrain` handler so informational logs,
  rating buttons and text input appear inside the window.
* `utils/logger.py` exposes `install_handler(handler)` so the GUI can hook into
  existing loggers without rewriting them. The new `set_stdout_level()` helper
  hides everything below ``WARNING`` on the terminal while the GUI is active.
* The run script `run_brain.sh` will accept `--gui_train` and forward it to
  `brain.py`.

## Future Improvements

* Visualise novelty metrics in additional panes. Hormone levels are now shown in
  the top bar of the training interface.
