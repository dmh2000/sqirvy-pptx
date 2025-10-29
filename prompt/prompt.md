# Prompts

Feel free to adapt these prompts as needed. 

## Prompt For Slide Notes
- You will be create a set of powerpoint slide notes. 
- Read the file "permuations.md" and create the slide notes as follows:
- Break the topic slide into logical sections. from those sections, create a markdown file with sections for each logical section. 
- label each section with a header "## Slide N : title". 
- add a description of each section. elaborate on the contents of each section if you can describe it in more detail. This description will be used to create text to speech audio so try to keep it to maximum 25 lines.
- delimit each section with the "---" break.
- write the results to the file permutation-notes.md

## Prompt For Slides

- You will create a set of powerpoint slides with text.
- Read the file permutation-notes.md and create the slide contents as follows:
- for each section in the permutation-notes.md, create a summary with a title and a list of bullet points for that section
- the list of bullet points should have no more than 8 lines
- delimit each slide with the "---" break.
- write the results to the file permutation-slides.md

# Prompt for Audio Bash Script
create a bash script that does the following
- find all files in the current directory of the form "audio-N.mp3" where N is the associated slide number. 
- sort the list in order of slide number
- for each file in the sorted list:
  - calls the function "embed_audio.py" for each file
    - first argument: -s [slide number]
    - second argument: the powerpoint file 
    - third argument: the audio file
- write the bash script in cmd/embed_audio.sh

- Example files in "examples/permutation"

