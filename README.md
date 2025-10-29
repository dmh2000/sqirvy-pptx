# Sqirvy-pptx

Build a PowerPoint presentation from markdown and audio using AI.

## Steps To Product A PPTX Presentation

See [Permutations](#get-the-topic) for more details
1. Create, download, or find a file with the topic that you want to present. 
2. Create the presentation slide notes markdown file using AI
3. Create the presentation slides file by using AI
4. Create she presentation PowerPoint presentation
5. Creat the audio narration
6. embed the audio narration into the presentation
 

## Example 1 : Permutations

### Get the Topic

- Using Perplexity I asked for step by step algorithm to generate permuations. 

```text
To generate all permutations of a list or sequence, one of the most common step-by-step algorithms is recursive backtracking. Here’s a detailed step-by-step approach suitable for both manual understanding and programming implementation:

Recursive Backtracking Algorithm
Start with an Input List

Have a sequence of elements (e.g., [1][2][3]).

Define a Function to Permute

The function takes the current index (start) and recursively swaps every other index (i) from start to the end of the list.

Base Case

If start reaches the end of the list (e.g., start == len(list) - 1), record the current arrangement as a permutation.

Recursive Step (for-loop and swapping)

Loop i from start to len(list) - 1:

Swap elements at indices start and i

Recursively call the function with start + 1

Swap back (backtrack) to restore original ordering before repeating with the next i.

Repeat

This process explores all possible placements for each element position, producing every unique arrangement.
```

### Create Slide Notes

- Using the AI of your choice, prompt it to create the slide notes

```markdown
- You will be create a set of powerpoint slide notes. 
- Read the file "permuations.md" and create the slide notes as follows:
- Break the topic slide into logical sections. from those sections, create a markdown file with sections for each logical section. 
- label each section with a header "## Slide N : title". 
- add a description of each section. elaborate on the contents of each section if you can describe it in more detail. This description will be used to create text to speech audio so try to keep it to maximum 25 lines.
- delimit each section with the "---" break.
- write the results to the file permutation-notes.md
```

### Create Slides

- Using the AI of your choice, prompt it to create the individual slides

```markdown
- You will create a set of powerpoint slides with text.
- Read the file permutation-notes.md and create the slide contents as follows:
- for each section in the permutation-notes.md, create a summary with a title and a list of bullet points for that section
- the list of bullet points should have no more than 8 lines
- delimit each slide with the "---" break.
- write the results to the file permutation-slides.md
```

### Create the PowerPoint Presentation
- Use the function "md_to_pptx" to create the PowerPoint presentation

```bash
>python md_to_pptx.py <slides.md> [notes.md]
  Usage: python md_to_pptx.py <input.md> [notes.md]

>python md_to_pptx.py permutation-slides.md permutation-notes.md 
  Reading permutation-slides.md...
  Found 6 slides
  Reading notes from permutation-notes.md...
  Found 6 note sections
  Creating PowerPoint presentation...
  PowerPoint presentation saved to: permutation-slides.pptx
  Done!

```

### Create Audio Narration MP3 Files

- Use the function 'convert_notes_to_audio' to convert the slide notes into audio 
- This function will output one mp3 audio file for each section in the slide notes

```bash

> python convert-notes-to-audio.py permutation-notes.md 
audio-1.mp3
audio-3.mp3
audio-2.mp3
audio-4.mp3
audio-5.mp3
audio-6.mp3
```

### Embed the Audio into the slides