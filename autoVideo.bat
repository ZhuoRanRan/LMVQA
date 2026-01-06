

@echo off
REM Loop through numbers 6 to 10
for %%i in (4) do (
    REM Run the first Python script
	REM echo Running: videoname %%i.mp4 step 1
	REM python .\main_gpt4o.py --video .\Lecture_Video\Lecture%%i.mp4 --question "Please describe this video"

    REM Run the second Python script
	echo Running: videoname %%i.mp4 step 2

	python .\qa_generate_gpt4o.py "Lecture_Video/Lecture%%i.mp4"

	REM Run the third Python script
	echo Running: videoname %%i.mp4 step 3

	python .\eval_main.py --video_name Lecture%%i
)
for %%i in (5) do (
    REM Run the first Python script
	REM echo Running: videoname %%i.mp4 step 1
	python .\main_gpt4o.py --video .\Lecture_Video\Lecture%%i.mp4 --question "Please describe this video"

    REM Run the second Python script
	echo Running: videoname %%i.mp4 step 2

	python .\qa_generate_gpt4o.py "Lecture_Video/Lecture%%i.mp4"

	REM Run the third Python script
	echo Running: videoname %%i.mp4 step 3

	python .\eval_main.py --video_name Lecture%%i
)