# AGENTS.md

# Basic Instructions

1. Please refer frequently to `brain/docs/human_brain_components_reference.txt`. It serves as the
reference standard of which we are trying to build around.

2. Any model generated or trained should be preserved and saved to the 'persistent' dir

3. Whenever you add a new package, update the requirements.txt

4. Whenever usage instructions change, update README.md

5. If you need a new bootstrap model for any portion of brain's systems, add the download 
functionality to models/model_initialization_scripts/download_models.py

6. Refer to neurosymbolic_plan.md for the latest instructions for what you should be doing
to continue the development process. As you uncover new improvements which can be made, add
them to neurosymbolic_plan.md such that it can be addressed in the next PR

7. Do not name any scripts, models, etc. anything ambiguous. It is okay for a name to be a word or two long, perhaps even longer, as long as it is unmistakable what the intention of it is.