# &#x20;**Vision-Based Desktop Automation with Dynamic Icon Grounding**



## **Objective**



This project automates Windows Notepad using vision-based desktop grounding.



The system dynamically locates the Notepad desktop shortcut regardless of its position on the Windows desktop, opens Notepad, writes posts fetched from the JSONPlaceholder API, saves each post as a text file, closes Notepad, and repeats the process.



The assignment intentionally uses a vision-heavy approach even though Notepad could be launched more simply. The goal is to demonstrate scalable GUI grounding, not task-specific shortcuts.



\---



## **Features**



* Planner-guided VLM grounding using Gemini Vision
* Candidate-region proposal from full desktop screenshots
* Grounder refinement inside cropped candidate regions
* Dynamic icon detection regardless of icon position
* Handles cluttered desktops with many icons
* Supports different icon placements such as top-left, center, and bottom-right
* Expands tight planner regions to preserve icon and label context
* Uses cached coordinates after first successful detection to reduce repeated VLM calls
* Locally validates cached coordinates using pixel similarity
* Direct full-screen VLM fallback
* Optional OpenCV template-matching fallback
* Saves annotated screenshots of detections
* Verifies all generated output files



\---



## **Requirements**



* Windows 10/11
* 1920x1080 resolution recommended
* Python 3.10+
* Notepad shortcut visible on the desktop
* Gemini API key stored as an environment variable



\---



## **Setup**



Install `uv`:



```powershell

pip install uv



Install project dependencies:



uv sync



Set your Gemini API key:



$env:GEMINI\_API\_KEY="your\_real\_gemini\_api\_key\_here"



Run the automation:



uv run vision-automation



Alternative run command:



uv run python -m vision\_desktop\_automation.main



## **Automation Workflow**

Show the desktop.

Capture a fresh screenshot.

Locate the Notepad desktop shortcut using planner-guided VLM grounding.

Double-click the detected icon coordinates.

Validate that Notepad launched.

Fetch the first 10 posts from JSONPlaceholder.

Paste each post into Notepad using this format:

Title: {title}



{body}

Save each post as post\_{id}.txt.

Store the files inside Desktop/tjm-project.

Close Notepad.

Repeat the process for all 10 posts.

Verify that all generated files exist and contain the expected title.



## **Grounding Approach**



The primary grounding method is inspired by ScreenSeekeR-style cascaded visual search.



The system separates the visual search process into two main roles:



#### 1\. Planner



The planner receives the full desktop screenshot and proposes candidate regions where the Notepad icon is likely to exist.



Instead of asking the model to directly click from the full screenshot, the planner narrows the search area first.



#### 2\. Grounder



The grounder receives a cropped candidate region and returns:



A bounding box around the likely target

A click point at the icon center

Confidence values

Label and visual match scores



This makes the system more flexible than hardcoded coordinates or simple template matching.



#### Why Planner-Guided Grounding?



Hardcoded coordinates fail when the icon moves.



Template matching can fail when:



The icon size changes

The theme changes

The desktop background is busy

The shortcut label changes

The icon is slightly different across Windows versions



The VLM-based planner-grounder approach is more general because it can reason about both:



The icon appearance

The text label



This makes the solution closer to a general GUI grounding system rather than a task-specific Notepad launcher.



#### Fallback Strategy



The fallback hierarchy is:



Cached coordinates if locally valid

Planner-guided VLM grounding

Direct full-screen VLM grounding

Template matching fallback if local templates are available

Clean failure with diagnostic screenshot and logs



Template matching is included only as graceful degradation. The primary method remains planner-guided VLM grounding.



## **Cache Strategy**



After the first successful detection, the system stores the Notepad coordinates and a small reference crop around the icon.



For later posts, the system checks whether the icon is still at the cached location using local pixel similarity.



If the cache is valid, the system reuses the stored coordinates instead of calling Gemini again. This reduces:



API calls

Runtime

Risk of Gemini 503 errors

Cost and latency



If the cache is invalid, the system reruns visual grounding.



## **Error Handling**



The system includes handling for:



Icon not found

Gemini API timeout or 503 errors

Invalid or partial JSON returned by the VLM

Save dialog delays

Existing output files

Notepad launch validation

Leftover Notepad windows

Failed saves

Missing or empty output files



The script also saves debug screenshots in failure\_screenshots/ when failures occur.



## **Detection Examples**

* Top-left annotated detection



* Top-left popup confirmation



* Center annotated detection



* Center popup confirmation



* Bottom-left annotated detection 1



* Bottom-left annotated detection 2



* Bottom-right popup confirmation



## **Output**



Generated files are saved to:



Desktop/tjm-project/



Expected output:



post\_1.txt

post\_2.txt

post\_3.txt

post\_4.txt

post\_5.txt

post\_6.txt

post\_7.txt

post\_8.txt

post\_9.txt

post\_10.txt



Each file contains:



Title: {title}



{body}

## **Project Structure**

vision-desktop-automation/

│

├── README.md

├── pyproject.toml

├── uv.lock

├── .gitignore

├── .env.example

│

├── src/

│   └── vision\_desktop\_automation/

│       ├── \_\_init\_\_.py

│       └── main.py

│

├── docs/

│   └── screenshots/

│       ├── Bottom left 1 ANNOTATED screenshot.png

│       ├── Bottom left 2 ANNOTATED screenshot.png

│       ├── Bottom Right POP-UP screenshot.png

│       ├── Center ANNOTATED screenshot.png

│       ├── Center POP-UP screenshot.png

│       ├── Top left ANNOTATED screenshot.png

│       └── Top left POP-UP screenshot.png

│

├── templates/

│   └── .gitkeep

│

└── failure\_screenshots/

&#x20;   └── .gitkeep

## **Configuration**



The Gemini API key is not hardcoded in the source code.



The code reads it from the environment:



*GEMINI\_API\_KEY = os.getenv("GEMINI\_API\_KEY")*



Example PowerShell setup:



*$env:GEMINI\_API\_KEY="your\_real\_gemini\_api\_key\_here"*



A .env.example file is included only as a safe placeholder:



GEMINI\_API\_KEY=your\_gemini\_api\_key\_here



## **Known Limitations**

* First-time detection depends on Gemini API availability.
* Gemini may occasionally return 503 or timeout.
* Template matching fallback requires local template images in the templates/ folder.
* The project is designed for Windows desktop automation.
* The project was mainly tested on 1920x1080 resolution.
* Very cluttered desktops may increase grounding latency.
* If the icon is fully hidden behind another window, the system cannot visually ground it until the desktop is visible.

## **Future Improvements**

* Split the script into separate modules:
* grounding.py
* automation.py
* api.py
* utils.py
* Add CLI arguments for arbitrary target descriptions.
* Add support for grounding any desktop icon or GUI button.
* Add unit tests for JSON parsing and coordinate conversion.
* Improve screenshot annotation placement near screen edges.
* Add automatic screenshot naming by detected region.
* Add more robust local fallback methods for API outages.	

